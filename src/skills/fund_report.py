"""基金报告生成 Skill - 使用专业金融 LLM 生成分析报告

流程：
1. 将基金数据填充到英文模板
2. 搜索基金最新新闻 (通过 MCP Tavily)
3. 使用 finance-llama-8b 生成英文专业分析报告
4. 使用远程 Qwen API 将英文报告翻译成中文

支持并行处理：生成下一个基金报告的同时翻译当前报告
支持用户偏好注入：根据用户偏好调整分析风格和建议
支持 MCP 搜索：获取基金最新新闻和市场信息
"""
from typing import Any, Dict, Optional
from pathlib import Path
import json
import asyncio
import re

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from openai import AsyncOpenAI

from .base import BaseSkill, SkillOutput
from ..config import ollama_config, qwen_config
from ..storage.checkpoint import CheckpointManager
from ..utils.model_wrapper import PreferenceAwareModelMixin


class FundReportGeneratorSkill(BaseSkill, PreferenceAwareModelMixin):
    """基金报告生成 Skill - 调用专业金融模型生成投资分析报告

    支持用户偏好注入，根据用户设置调整分析风格和投资建议。
    支持 MCP 搜索，获取基金最新新闻和市场信息。
    """

    name = "fund_report_generator"
    description = "根据基金净值数据和技术指标，使用专业金融模型生成基金分析报告"

    # 模型配置
    FINANCE_MODEL = "martain7r/finance-llama-8b:q4_k_m"  # 金融分析模型 (本地 Ollama)

    # 翻译重试配置
    MAX_TRANSLATION_RETRIES = 3  # 最大重试次数
    MIN_CHINESE_RATIO = 0.3  # 中文字符最小比例 (30%)

    # 模板文件路径
    POSITION_ADVICE_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "fund_position_advice.md"

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        enable_mcp_search: bool = True,
    ):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager
        self._enable_mcp_search = enable_mcp_search
        self._mcp_search_skill = None
        # 初始化远程 Qwen 客户端
        self._qwen_client: Optional[AsyncOpenAI] = None

    async def _get_mcp_search_skill(self):
        """获取 MCP 搜索技能（懒加载）"""
        if not self._enable_mcp_search:
            return None

        if self._mcp_search_skill is None:
            try:
                from .mcp_search import MCPSearchSkill
                self._mcp_search_skill = MCPSearchSkill()
            except Exception as e:
                print(f"⚠️ MCP 搜索技能初始化失败: {e}")
                return None

        return self._mcp_search_skill

    async def _search_fund_news(self, fund_name: str, fund_code: str) -> str:
        """搜索基金最新新闻

        Args:
            fund_name: 基金名称
            fund_code: 基金代码

        Returns:
            新闻摘要文本，如果搜索失败返回空字符串
        """
        search_skill = await self._get_mcp_search_skill()
        if not search_skill:
            return ""

        try:
            print(f"🔍 [{fund_code}] 搜索最新新闻 (MCP Tavily)...")
            query = f"{fund_name} {fund_code} 基金 最新消息 持仓"
            result = await search_skill.execute(query, max_results=5, search_depth="basic")

            if result.success and result.result:
                print(f"✅ [{fund_code}] 获取到新闻信息")
                return result.result
            else:
                print(f"⚠️ [{fund_code}] 新闻搜索无结果")
                return ""

        except Exception as e:
            print(f"⚠️ [{fund_code}] 新闻搜索失败: {e}")
            return ""

    def _get_qwen_client(self) -> AsyncOpenAI:
        """获取 Qwen API 客户端（懒加载）"""
        if self._qwen_client is None:
            if not qwen_config.api_key:
                raise ValueError("QWEN_API_KEY 未配置，请在 .env 文件中设置")
            self._qwen_client = AsyncOpenAI(
                api_key=qwen_config.api_key,
                base_url=qwen_config.api_base,
                timeout=qwen_config.timeout,
            )
        return self._qwen_client

    def _is_chinese_text(self, text: str) -> bool:
        """检测文本是否包含足够比例的中文字符

        Args:
            text: 待检测文本

        Returns:
            True 如果中文字符比例 >= MIN_CHINESE_RATIO
        """
        if not text:
            return False

        # 移除 Markdown 格式符号、数字、标点符号等
        # 只保留中文字符和英文字母
        clean_text = re.sub(r'[#*|\-\[\](){}:，。！？、；：""''【】《》\s\d\.%+]', '', text)

        if len(clean_text) == 0:
            return False

        # 计算中文字符数量（CJK Unified Ideographs）
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))

        # 计算中文字符比例
        chinese_ratio = chinese_chars / len(clean_text)

        return chinese_ratio >= self.MIN_CHINESE_RATIO

    def _get_chinese_ratio(self, text: str) -> float:
        """计算文本中中文字符的比例

        Args:
            text: 待检测文本

        Returns:
            中文字符比例 (0.0 - 1.0)
        """
        if not text:
            return 0.0

        clean_text = re.sub(r'[#*|\-\[\](){}:，。！？、；：""''【】《》\s\d\.\%\+]', '', text)

        if len(clean_text) == 0:
            return 0.0

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))
        return chinese_chars / len(clean_text)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fund_data": {
                    "type": "string",
                    "description": "JSON格式的基金分析数据（由 fund_retriever 生成）"
                },
                "user_id": {
                    "type": "string",
                    "description": "用户标识符，用于获取用户偏好和持仓",
                    "default": "default"
                },
                "fund_code": {
                    "type": "string",
                    "description": "基金代码，用于获取该基金的持仓信息"
                }
            },
            "required": ["fund_data"]
        }

    async def _get_position_context(self, user_id: str, fund_code: Optional[str] = None) -> str:
        """获取用户持仓上下文

        Args:
            user_id: 用户标识
            fund_code: 基金代码

        Returns:
            格式化的持仓上下文字符串
        """
        if self._checkpoint_manager is None:
            return ""

        try:
            return await self._checkpoint_manager.get_positions_as_context(user_id, fund_code)
        except Exception as e:
            print(f"⚠️ 获取用户持仓失败: {e}")
            return ""

    def _load_position_advice_template(
        self,
        positions_context: str = "",
        prefs_context: str = "",
        current_nav: float = 0,
    ) -> str:
        """从文件加载持仓操作建议模板并填充变量

        Args:
            positions_context: 用户持仓上下文
            prefs_context: 用户偏好上下文
            current_nav: 当前净值

        Returns:
            填充后的模板内容
        """
        try:
            template_content = self.POSITION_ADVICE_TEMPLATE_PATH.read_text(encoding="utf-8")

            # 替换模板中的占位符
            result = template_content.format(
                positions_context=positions_context if positions_context else "无持仓信息",
                prefs_context=prefs_context if prefs_context else "无偏好信息",
                current_nav=current_nav,
            )

            return "\n" + result

        except FileNotFoundError:
            print(f"⚠️ 持仓建议模板文件不存在: {self.POSITION_ADVICE_TEMPLATE_PATH}")
            return ""
        except Exception as e:
            print(f"⚠️ 加载持仓建议模板失败: {e}")
            return ""

    async def execute(self, fund_data: str, user_id: str = "default", fund_code: Optional[str] = None, **kwargs) -> SkillOutput:
        """执行完整的基金报告生成流程（生成英文报告 + 翻译）

        Args:
            fund_data: 基金分析数据 JSON
            user_id: 用户标识符，用于获取偏好和持仓
            fund_code: 基金代码，用于获取该基金持仓
            **kwargs: 额外参数

        Returns:
            SkillOutput 包含生成的中文报告
        """
        try:
            # 1. 解析基金数据
            data = json.loads(fund_data)

            # 2. 获取用户偏好上下文
            prefs_context = await self.get_user_preferences_context(user_id, "fund")
            if prefs_context:
                print(f"📋 已加载用户偏好: {user_id}")

            # 3. 获取用户持仓上下文
            # 如果没有传入 fund_code，尝试从数据中获取
            if not fund_code:
                basic_info = data.get("basic_info", {})
                fund_code = basic_info.get("fund_code")

            positions_context = await self._get_position_context(user_id, fund_code)
            if positions_context:
                print(f"💼 已加载用户持仓: {user_id} / {fund_code}")

            # 4. 搜索最新新闻 (MCP Tavily)
            basic_info = data.get("basic_info", {})
            fund_name = basic_info.get("fund_name", "")
            fund_code = basic_info.get("fund_code", "")
            news_context = await self._search_fund_news(fund_name, fund_code)

            # 5. 生成英文报告（注入偏好、持仓和新闻）
            english_report = await self.generate_english_report(data, prefs_context, news_context, positions_context)

            # 6. 翻译成中文，并基于持仓和偏好添加个性化建议
            chinese_report = await self.translate_to_chinese(
                english_report,
                data,
                prefs_context=prefs_context,
                positions_context=positions_context,
            )

            return SkillOutput(
                success=True,
                result=chinese_report
            )

        except json.JSONDecodeError as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"基金数据格式错误: {str(e)}"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SkillOutput(
                success=False,
                result=None,
                error=f"报告生成失败: {str(e)}"
            )

    async def generate_english_report(
        self,
        data: Dict[str, Any],
        prefs_context: str = "",
        news_context: str = "",
        positions_context: str = ""
    ) -> str:
        """生成英文分析报告（使用本地 Ollama finance-llama）

        Args:
            data: 基金分析数据字典
            prefs_context: 用户偏好上下文
            news_context: 最新新闻上下文 (来自 MCP 搜索)
            positions_context: 用户持仓上下文

        Returns:
            英文分析报告
        """
        basic_info = data.get("basic_info", {})
        fund_code = basic_info.get("fund_code", "")
        fund_name = basic_info.get("fund_name", "")

        # 1. 填充英文模板
        print(f"📝 [{fund_code}] 填充英文基金分析模板...")
        english_template = self._fill_english_template(data)

        # 2. 构建英文 Prompt 并调用金融模型
        print(f"🤖 [{fund_code}] 调用金融模型 ({self.FINANCE_MODEL}) 生成英文分析...")
        english_prompt = self._build_english_prompt(data, english_template, prefs_context, news_context, positions_context)
        english_report = await self._call_finance_llm(english_prompt)

        print(f"✅ [{fund_code}] 英文报告生成完成 ({len(english_report)} 字符)")
        return english_report

    async def translate_to_chinese(
        self,
        english_report: str,
        data: Dict[str, Any],
        prefs_context: str = "",
        positions_context: str = "",
    ) -> str:
        """将英文报告翻译成中文，并基于持仓和偏好添加个性化建议（使用远程 Qwen API）

        包含重试机制：如果翻译失败或返回内容不是中文，最多重试 MAX_TRANSLATION_RETRIES 次

        Args:
            english_report: 英文报告内容
            data: 基金分析数据字典
            prefs_context: 用户偏好上下文
            positions_context: 用户持仓上下文

        Returns:
            中文翻译报告（包含个性化持仓建议）
        """
        basic_info = data.get("basic_info", {})
        fund_code = basic_info.get("fund_code", "")
        fund_name = basic_info.get("fund_name", "")
        latest_nav = data.get("latest_nav", {})
        current_nav = latest_nav.get("nav", 0)

        # 构建个性化建议部分的提示
        personalization_prompt = ""
        if positions_context or prefs_context:
            personalization_prompt = self._load_position_advice_template(
                positions_context=positions_context,
                prefs_context=prefs_context,
                current_nav=current_nav,
            )

        prompt = f"""你是一位专业的基金投资顾问。请完成以下任务：

## 任务1：翻译报告
将以下英文基金分析报告翻译成中文。

翻译要求：
1. 保持专业金融术语的准确性
2. 语言流畅自然，符合中文表达习惯
3. 保留原报告的结构和格式（标题、列表、表格等）
4. 关键结论和操作建议要**加粗**突出显示
5. 净值、涨跌幅、收益率等数字保持不变
6. 基金名称使用: {fund_name} ({fund_code})
7. 使用适当的 emoji 增强可读性：
   - 📊 当日估值
   - 📈 趋势判断
   - 📉 业绩回顾
   - 🎯 技术面总结
   - 💡 操作建议
{personalization_prompt}
## 英文报告原文
---
{english_report}
---

## 输出要求
1. 先输出翻译后的中文报告
2. {"将 '💼 持仓操作建议' 章节放在 '📊 当日估值' 之后（不是报告最后）" if positions_context else ""}
3. 必须输出中文内容，不要输出英文
4. 保持 Markdown 格式

请输出完整报告：
"""

        last_error = None
        last_result = None

        for attempt in range(1, self.MAX_TRANSLATION_RETRIES + 1):
            try:
                print(f"🔄 [{fund_code}] 调用远程 Qwen API ({qwen_config.model}) 翻译成中文 (尝试 {attempt}/{self.MAX_TRANSLATION_RETRIES})...")

                client = self._get_qwen_client()
                response = await client.chat.completions.create(
                    model=qwen_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=qwen_config.temperature,
                )
                result = response.choices[0].message.content

                # 检查是否返回空内容
                if not result or len(result.strip()) < 50:
                    print(f"⚠️ [{fund_code}] 第 {attempt} 次翻译返回内容过短或为空")
                    last_error = "返回内容过短"
                    last_result = result
                    continue

                # 检查返回内容是否为中文
                chinese_ratio = self._get_chinese_ratio(result)
                if not self._is_chinese_text(result):
                    print(f"⚠️ [{fund_code}] 第 {attempt} 次翻译返回非中文内容 (中文比例: {chinese_ratio:.1%})")
                    last_error = f"中文比例过低: {chinese_ratio:.1%}"
                    last_result = result
                    continue

                # 翻译成功
                print(f"✅ [{fund_code}] 中文翻译完成 ({len(result)} 字符, 中文比例: {chinese_ratio:.1%})")
                return result

            except Exception as e:
                print(f"⚠️ [{fund_code}] 第 {attempt} 次翻译调用失败: {e}")
                last_error = str(e)
                # 等待一小段时间后重试
                if attempt < self.MAX_TRANSLATION_RETRIES:
                    await asyncio.sleep(1)

        # 所有重试都失败
        print(f"❌ [{fund_code}] 翻译失败，已重试 {self.MAX_TRANSLATION_RETRIES} 次，使用英文原文")
        print(f"   最后错误: {last_error}")

        # 如果有部分翻译结果，尝试使用它
        if last_result and len(last_result.strip()) >= 50:
            return f"# {fund_name} ({fund_code}) 分析报告\n\n[部分翻译，可能包含英文]\n\n{last_result}"

        return f"[翻译失败: {last_error}]\n\n# {fund_name} ({fund_code}) Analysis Report\n\n{english_report}"

    def _fill_english_template(self, data: Dict[str, Any]) -> str:
        """填充英文基金分析模板"""
        basic_info = data.get("basic_info", {})
        latest_nav = data.get("latest_nav", {})
        trend = data.get("trend", {})
        rsi = data.get("rsi", {})
        macd = data.get("macd", {})
        kdj = data.get("kdj", {})
        boll = data.get("bollinger", {})
        risk_metrics = data.get("risk_metrics", {})
        history = data.get("history", [])

        # 格式化历史净值
        history_str = "\n".join([
            f"| {h['date']} | {h['nav']} | {h['pct_change']}% |"
            for h in history[-10:]
        ])

        # 格式化区间收益
        returns = trend.get("returns", {})
        returns_rows = []
        return_labels = {
            "1w": "1 Week",
            "1m": "1 Month",
            "3m": "3 Months",
            "6m": "6 Months",
            "1y": "1 Year",
            "ytd": "Year to Date"
        }
        for k, v in returns.items():
            if v is not None:
                label = return_labels.get(k, k)
                returns_rows.append(f"| {label} | {v}% |")
        returns_str = "\n".join(returns_rows)

        # 均线排列翻译
        ma_arrangement = trend.get("ma_arrangement", "N/A")
        ma_arrangement_en = {
            "多头排列": "Bullish Alignment (MA5 > MA10 > MA20 > MA60)",
            "空头排列": "Bearish Alignment (MA5 < MA10 < MA20 < MA60)",
            "交叉缠绕": "Mixed/Crossing",
            "数据不足": "Insufficient Data"
        }.get(ma_arrangement, ma_arrangement)

        # RSI 区域翻译
        rsi_zone = rsi.get("zone", "中性区")
        rsi_zone_en = {
            "超买区": "Overbought (>70)",
            "偏强区": "Bullish (60-70)",
            "超卖区": "Oversold (<30)",
            "偏弱区": "Bearish (30-40)",
            "中性区": "Neutral (40-60)"
        }.get(rsi_zone, rsi_zone)

        # MACD 信号翻译
        macd_signal = macd.get("signal", "无信号")
        macd_signal_en = {
            "金叉": "Golden Cross (Bullish)",
            "死叉": "Death Cross (Bearish)",
            "无信号": "No Signal"
        }.get(macd_signal, macd_signal)

        # KDJ 区域翻译
        kdj_zone = kdj.get("zone", "中性区")
        kdj_zone_en = {
            "超买区": "Overbought (K>80 or J>100)",
            "超卖区": "Oversold (K<20 or J<0)",
            "偏强区": "Bullish (K>50)",
            "偏弱区": "Bearish (K<50)",
            "中性区": "Neutral",
            "数据不足": "Insufficient Data"
        }.get(kdj_zone, kdj_zone)

        # 布林带位置翻译
        boll_position = boll.get("position", "N/A")
        if "上轨" in boll_position:
            boll_position_en = "Above Upper Band (Overbought)"
        elif "下轨" in boll_position:
            boll_position_en = "Below Lower Band (Oversold)"
        elif "中轨上方" in boll_position:
            boll_position_en = boll_position.replace("中轨上方", "Above Middle Band ")
        elif "中轨下方" in boll_position:
            boll_position_en = boll_position.replace("中轨下方", "Below Middle Band ")
        else:
            boll_position_en = boll_position

        # 基金类型翻译
        fund_type = basic_info.get("fund_type", "N/A")
        fund_type_en = {
            "股票型": "Equity Fund",
            "混合型": "Hybrid Fund",
            "债券型": "Bond Fund",
            "货币型": "Money Market Fund",
            "指数型": "Index Fund",
            "QDII": "QDII Fund",
            "FOF": "Fund of Funds"
        }.get(fund_type, fund_type)

        ma_values = trend.get("ma_values", {})

        # 夏普比率评级
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        if sharpe >= 2:
            sharpe_rating = "Excellent (>=2)"
        elif sharpe >= 1:
            sharpe_rating = "Good (1-2)"
        elif sharpe >= 0.5:
            sharpe_rating = "Fair (0.5-1)"
        else:
            sharpe_rating = "Poor (<0.5)"

        # 连涨/连跌翻译
        streak = risk_metrics.get("current_streak", {})
        streak_type = streak.get("type", "无")
        streak_days = streak.get("days", 0)
        streak_en = {
            "涨": "Up",
            "跌": "Down",
            "平": "Flat",
            "无": "N/A"
        }.get(streak_type, streak_type)

        template = f"""# Fund Technical Analysis Report

## Basic Information
| Item | Value |
|------|-------|
| Fund Code | {basic_info.get('fund_code', 'N/A')} |
| Fund Name | {basic_info.get('fund_name', 'N/A')} |
| Fund Type | {fund_type_en} |
| Fund Manager | {basic_info.get('fund_manager', 'N/A')} |
| Report Date | {latest_nav.get('date', 'N/A')} |

## Latest NAV (Net Asset Value)
| Metric | Value |
|--------|-------|
| NAV | {latest_nav.get('nav', 'N/A')} |
| Daily Change | {latest_nav.get('pct_change', 0)}% |
| Accumulated NAV | {latest_nav.get('acc_nav', 'N/A')} |

## Performance Returns
| Period | Return |
|--------|--------|
{returns_str}

## Risk-Return Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Annualized Return | {risk_metrics.get('annual_return', 'N/A')}% | Fund's yearly return |
| Annualized Volatility | {risk_metrics.get('annual_volatility', 'N/A')}% | Price fluctuation |
| Sharpe Ratio | {risk_metrics.get('sharpe_ratio', 'N/A')} | {sharpe_rating} |
| Calmar Ratio | {risk_metrics.get('calmar_ratio', 'N/A')} | Return per unit drawdown |
| Sortino Ratio | {risk_metrics.get('sortino_ratio', 'N/A')} | Downside risk adjusted |
| Max Drawdown | {trend.get('max_drawdown', 0)}% | Maximum peak-to-trough decline |
| Win Rate | {risk_metrics.get('win_rate', 'N/A')}% | Percentage of up days |
| Profit/Loss Ratio | {risk_metrics.get('profit_loss_ratio', 'N/A')} | Avg gain vs avg loss |
| Current Streak | {streak_days} days {streak_en} | Momentum indicator |

## Moving Averages
| MA | Value |
|----|-------|
| MA5 | {ma_values.get('ma5', 'N/A')} |
| MA10 | {ma_values.get('ma10', 'N/A')} |
| MA20 | {ma_values.get('ma20', 'N/A')} |
| MA60 | {ma_values.get('ma60', 'N/A')} |

- **MA Arrangement**: {ma_arrangement_en}
- **Price vs MA**: NAV is {"above" if latest_nav.get('nav', 0) > ma_values.get('ma20', 0) else "below"} MA20

## MACD Indicator
| Metric | Value |
|--------|-------|
| DIF (Fast Line) | {macd.get('dif', 'N/A')} |
| DEA (Signal Line) | {macd.get('dea', 'N/A')} |
| MACD Histogram | {macd.get('macd', 'N/A')} |
| EMA12 | {macd.get('ema12', 'N/A')} |
| EMA26 | {macd.get('ema26', 'N/A')} |
| Signal | **{macd_signal_en}** |

## KDJ Indicator
| Metric | Value |
|--------|-------|
| K | {kdj.get('k', 'N/A')} |
| D | {kdj.get('d', 'N/A')} |
| J | {kdj.get('j', 'N/A')} |
| Zone | **{kdj_zone_en}** |

## RSI Indicator
| Metric | Value |
|--------|-------|
| RSI (14) | {rsi.get('rsi', 50)} |
| Zone | {rsi_zone_en} |

## Bollinger Bands
| Band | Value |
|------|-------|
| Upper Band | {boll.get('upper', 'N/A')} |
| Middle Band | {boll.get('mid', 'N/A')} |
| Lower Band | {boll.get('lower', 'N/A')} |
| Band Width | {boll.get('width', 'N/A')}% |
| Position | {boll_position_en} |

## NAV History (Last 10 Days)
| Date | NAV | Change |
|------|-----|--------|
{history_str}
"""
        return template

    def _build_english_prompt(
        self,
        data: Dict[str, Any],
        filled_template: str,
        prefs_context: str = "",
        news_context: str = "",
        positions_context: str = ""
    ) -> str:
        """构建英文分析提示词

        Args:
            data: 基金分析数据
            filled_template: 填充后的英文模板
            prefs_context: 用户偏好上下文
            news_context: 最新新闻上下文
            positions_context: 用户持仓上下文

        Returns:
            完整的英文 prompt
        """
        basic_info = data.get("basic_info", {})
        latest_nav = data.get("latest_nav", {})
        trend = data.get("trend", {})

        # 构建偏好指导部分
        preference_guidance = ""
        if prefs_context:
            preference_guidance = f"""

---
**IMPORTANT: User Preferences**

The following are the user's investment preferences. Please tailor your analysis and recommendations accordingly:

{prefs_context}

When making recommendations:
- If risk_tolerance is "conservative": Focus on capital preservation, recommend bond funds or money market funds, emphasize downside protection
- If risk_tolerance is "aggressive": Can suggest equity funds and higher-risk options, but still include risk assessment
- Consider fund_type preferences when evaluating fund suitability
- **CRITICAL - investment_horizon preference**:
  - If "short-term": Focus on liquidity, recent volatility, short-term NAV trends, suitable for money market or short-term bond funds
  - If "medium-term": Balance between growth and stability, focus on 3-6 month performance, sector rotation opportunities
  - If "long-term": Emphasize long-term returns, compound growth, fund manager track record, expense ratios, suitable for equity funds or balanced funds with DCA (定投) strategy
- If dividend_preference is specified, include relevant dividend/distribution analysis

---
"""

        # 构建持仓上下文部分
        positions_guidance = ""
        if positions_context:
            positions_guidance = f"""

---
**User's Current Position in This Fund**

{positions_context}

**IMPORTANT - Personalized Analysis Based on User's Position:**
Please incorporate the user's position into your analysis and recommendations:

1. **If user is holding this fund:**
   - Calculate and mention estimated current value (shares × current NAV)
   - Calculate and mention unrealized P&L (current value - cost amount)
   - Calculate and mention P&L percentage ((current NAV / cost NAV - 1) × 100%)

2. **If user has profit (current NAV > cost NAV):**
   - Discuss profit-taking strategies if appropriate
   - Suggest whether to lock in gains or let profits run based on technical signals
   - Consider their investment horizon when making recommendations

3. **If user has loss (current NAV < cost NAV):**
   - Discuss cost averaging (加仓摊薄成本) opportunities if technical signals support
   - Evaluate if the loss is temporary (fund fundamentals intact) or structural
   - Suggest whether to hold, average down, or cut losses based on analysis

4. **Position-aware recommendations:**
   - "Add position" recommendations should consider their current cost basis
   - "Reduce position" should consider their P&L situation
   - Always reference their specific holding period (from first_buy_date)

---
"""

        # 构建新闻上下文部分
        news_section = ""
        if news_context:
            news_section = f"""

---
**Recent News and Market Information** (from web search):

{news_context[:2000]}

Please incorporate relevant news and market information into your analysis, especially:
- Fund holdings or strategy changes
- Fund manager changes or comments
- Market trends affecting this fund category
- Regulatory changes affecting the fund

---
"""

        prompt = f"""You are a senior fund investment analyst with over 15 years of experience in Chinese mutual fund market. You are an expert in fund analysis, including NAV trends, technical indicators (MACD, KDJ, RSI, Bollinger Bands), and risk-adjusted performance metrics (Sharpe Ratio, Calmar Ratio, Sortino Ratio).
{preference_guidance}{positions_guidance}{news_section}

**IMPORTANT ANALYSIS GUIDELINES:**

## Technical Indicator Analysis:
1. **MACD Analysis**:
   - DIF > DEA = Bullish momentum
   - Golden Cross (DIF crosses above DEA) = Buy signal
   - Death Cross (DIF crosses below DEA) = Sell signal
   - MACD Histogram expansion = Trend strengthening
   - MACD Histogram contraction = Trend weakening

2. **KDJ Analysis**:
   - K > 80 or J > 100 = Overbought zone (potential reversal)
   - K < 20 or J < 0 = Oversold zone (potential bounce)
   - Golden Cross (K crosses above D) in low zone = Strong buy signal
   - Death Cross (K crosses below D) in high zone = Strong sell signal
   - J line is most sensitive, use for early signals

3. **RSI Analysis**:
   - RSI > 70 = Overbought, potential reversal
   - RSI < 30 = Oversold, potential bounce
   - RSI divergence with price = Important reversal signal
   - RSI crossing 50 = Trend confirmation

4. **Bollinger Bands**:
   - Price touching upper band = Potential resistance
   - Price touching lower band = Potential support
   - Band width narrowing = Expect volatility expansion (squeeze)
   - Band width widening = Trend acceleration

## Risk-Return Analysis:
- **Annualized Return**: Fund's yearly return rate
  - Compare with benchmark (CSI 300: ~8-10%, bond funds: ~3-5%)
  - >15% for equity funds is good, >20% is excellent
- **Annualized Volatility**: Price fluctuation intensity
  - <10% = Low volatility (bond funds typical)
  - 10-20% = Medium volatility (balanced funds)
  - >20% = High volatility (equity funds typical)
  - Compare with fund type expectations
- **Sharpe Ratio**: Risk-adjusted return (excess return per unit of total risk)
  - >2 = Excellent
  - 1-2 = Good
  - 0.5-1 = Fair
  - <0.5 = Poor, needs attention
- **Calmar Ratio**: Return per unit of max drawdown (higher = better drawdown control)
  - >3 = Excellent risk management
  - 1-3 = Good
  - <1 = Drawdown too large relative to return
- **Sortino Ratio**: Downside risk-adjusted return (only considers negative volatility)
  - Compare with Sharpe: Sortino > Sharpe means less downside volatility (good)
  - Sortino < Sharpe means more downside volatility (caution)
- **Win Rate**: Percentage of positive return days
  - >55% = Good consistency
  - 45-55% = Normal
  - <45% = Relies on big wins to compensate
- **Profit/Loss Ratio**: Average gain / Average loss
  - >1.5 = Good risk-reward
  - 1-1.5 = Fair
  - <1 = Losses larger than gains on average (problematic)
- **Win Rate + P/L Ratio Combined**:
  - High win rate + High P/L ratio = Ideal
  - Low win rate + High P/L ratio = Trend-following style (acceptable)
  - High win rate + Low P/L ratio = Mean-reversion style (acceptable)
  - Low win rate + Low P/L ratio = Problematic
- **Current Streak**: Momentum indicator
  - Extended winning streak (>5 days) may signal overbought, watch for reversal
  - Extended losing streak (>5 days) may signal oversold, watch for bounce

Please analyze the following fund's data and generate a professional daily investment brief.

**Fund**: {basic_info.get('fund_name', 'N/A')} ({basic_info.get('fund_code', 'N/A')})
**Current NAV**: {latest_nav.get('nav', 'N/A')}
**Daily Change**: {latest_nav.get('pct_change', 0)}%
**Fund Type**: {basic_info.get('fund_type', 'N/A')}

---

Below is the complete fund analysis data:

{filled_template}

---

**Please generate a CONCISE daily investment brief (suitable for quick reading) covering:**

1. **Today's NAV Assessment**
   - Summarize today's NAV and change
   - Highlight any abnormal movements (>±2%)

2. **Trend Analysis** (MUST use one of these tags)
   - Clear trend judgment: **[UPTREND]** / **[CONSOLIDATION]** / **[DOWNTREND]**
   - MA system alignment analysis
   - MACD trend direction and any recent crossover signals
   - KDJ position and potential crossover signals

3. **Performance Review**
   - Brief analysis of recent period returns
   - Compare with benchmark if significant

4. **Risk-Return Profile**
   - Sharpe Ratio assessment (good/fair/poor)
   - Calmar and Sortino interpretation
   - Max drawdown significance
   - Win rate and streak analysis

5. **News Impact Analysis** (if recent news is available)
   - How recent news may affect the fund
   - Any material events to monitor

6. **Technical Summary**
   - Combined reading of MACD + KDJ + RSI + Bollinger
   - Identify confluences (multiple indicators agreeing)
   - Note any divergences or conflicting signals

7. **Investment Recommendation** (MUST use one of these tags)
   - **Action**: **[BUY]** / **[ADD]** / **[HOLD]** / **[REDUCE]** / **[WAIT]** / **[SELL]**
   - **Strategy**: Specific action plan with entry/exit levels
   - **Key Levels**: Support levels (from Bollinger lower, MA support) and resistance levels (from Bollinger upper, MA resistance)
   - **Risk Management**: Stop-loss suggestion based on max drawdown and volatility

**Format Requirements:**
- Use clear headings and bullet points
- Bold all key numbers and conclusions
- Keep total length around 400-600 words
- Use professional but accessible language
- Do NOT include disclaimers or risk warnings
"""
        return prompt

    async def _call_finance_llm(self, prompt: str) -> str:
        """调用金融专业模型生成英文报告"""
        try:
            llm = ChatOllama(
                model=self.FINANCE_MODEL,
                base_url=ollama_config.base_url,
                temperature=0.7,
            )

            response = await llm.ainvoke(prompt)
            return response.content

        except Exception as e:
            raise Exception(f"金融模型调用失败: {str(e)}")

    async def close(self) -> None:
        """关闭资源，包括 MCP 客户端

        确保 MCP 搜索技能正确关闭，防止 cancel scope 跨任务错误。
        """
        if self._mcp_search_skill is not None:
            try:
                await self._mcp_search_skill.close()
            except Exception as e:
                print(f"⚠️ 关闭 MCP 搜索技能失败: {e}")
            finally:
                self._mcp_search_skill = None
