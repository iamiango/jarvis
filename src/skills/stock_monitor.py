"""股票监控 Skill - 盘中定时监控并发送买卖提醒

工作流程：
1. 获取股票原始行情数据 (AKShare)
2. 使用本地 Finance LLM 分析技术指标
3. 获取用户持仓和偏好
4. 发送分析报告 + 持仓 + 偏好给远端 Qwen 模型
5. Qwen 给出结构化操作建议（持有/买入/卖出）
6. 若为买入/卖出建议，整理成邮件并发送

结构化输出格式：
{
    "stock_code": "600588",
    "stock_name": "用友网络",
    "current_price": 27.42,
    "action": "买入" | "持有" | "卖出",
    "reason": "操作理由",
    "position_info": {
        "shares": 1000,
        "cost_price": 31.04,
        "profit_loss_pct": -11.66
    }
}
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from openai import AsyncOpenAI

from .base import BaseSkill, SkillOutput
from .stock_data_fetcher import (
    fetch_stock_raw_data,
    build_finance_llm_prompt,
)
from ..config import ollama_config, qwen_config
from ..storage.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# Prompt 文件目录
PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_template(filename: str) -> str:
    """从 md 文件加载 prompt 模板"""
    prompt_path = PROMPTS_DIR / filename
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    else:
        logger.warning(f"Prompt 文件不存在: {prompt_path}")
        return ""


class TradeAction(str, Enum):
    """交易操作类型"""
    HOLD = "持有"
    BUY = "买入"
    SELL = "卖出"
    OPEN_POSITION = "建仓"


@dataclass
class StockAdvice:
    """股票操作建议"""
    stock_code: str
    stock_name: str
    current_price: float
    action: TradeAction
    reason: str
    position_shares: Optional[float] = None
    cost_price: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    analysis_summary: str = ""
    llm_request: str = ""  # 发送给远端 LLM 的请求报文

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "current_price": self.current_price,
            "action": self.action.value,
            "reason": self.reason,
            "position_info": {
                "shares": self.position_shares,
                "cost_price": self.cost_price,
                "profit_loss_pct": self.profit_loss_pct,
            } if self.position_shares else None,
            "analysis_summary": self.analysis_summary,
            "llm_request": self.llm_request,
        }


class StockMonitorSkill(BaseSkill):
    """股票监控 Skill - 盘中监控并给出操作建议

    特点：
    - 结合技术分析报告、用户持仓、风险偏好
    - 输出结构化操作建议
    - 仅在买入/卖出时触发邮件提醒
    """

    name = "stock_monitor"
    description = "监控股票并给出结构化操作建议（持有/买入/卖出）"

    # 模型配置
    FINANCE_MODEL = "martain7r/finance-llama-8b:q4_k_m"  # 本地金融分析模型

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager
        self._qwen_client: Optional[AsyncOpenAI] = None

    def _get_qwen_client(self) -> AsyncOpenAI:
        """获取 Qwen API 客户端（懒加载）"""
        if self._qwen_client is None:
            if not qwen_config.api_key:
                raise ValueError("QWEN_API_KEY 未配置")
            self._qwen_client = AsyncOpenAI(
                api_key=qwen_config.api_key,
                base_url=qwen_config.api_base,
                timeout=qwen_config.timeout,
                max_retries=qwen_config.max_retries,
            )
        return self._qwen_client

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "stock_code": {
                    "type": "string",
                    "description": "股票代码（6位数字）"
                },
                "user_id": {
                    "type": "string",
                    "description": "用户标识符",
                    "default": "default"
                }
            },
            "required": ["stock_code"]
        }

    async def execute(
        self,
        stock_code: str,
        user_id: str = "default",
        **kwargs
    ) -> SkillOutput:
        """执行股票监控分析

        Args:
            stock_code: 股票代码
            user_id: 用户标识

        Returns:
            SkillOutput 包含 StockAdvice 结构化结果
        """
        try:
            logger.info(f"🔍 [{stock_code}] 开始监控分析...")

            # 1. 获取股票数据
            stock_data = await self._get_stock_data(stock_code)
            if not stock_data:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"获取股票 {stock_code} 数据失败"
                )

            stock_name = stock_data.get("basic_info", {}).get("name", stock_code)
            current_price = stock_data.get("latest_quote", {}).get("close", 0)

            # 2. 使用本地 Finance LLM 生成分析
            logger.info(f"🤖 [{stock_code}] 调用本地 Finance LLM 分析...")
            analysis_report = await self._generate_analysis(stock_data)

            # 3. 获取用户持仓
            position_context = ""
            position_data = None
            if self._checkpoint_manager:
                position_data = await self._checkpoint_manager.get_stock_position(user_id, stock_code)
                if position_data and position_data.get("shares", 0) > 0:
                    position_context = self._format_position_context(position_data, current_price)
                    logger.info(f"💼 [{stock_code}] 已加载持仓信息")
                else:
                    position_context = "## 用户持仓信息\n当前无持仓，请根据技术面分析判断是否为合适的建仓时机。"
                    logger.info(f"💼 [{stock_code}] 当前无持仓")
            else:
                position_context = "## 用户持仓信息\n当前无持仓，请根据技术面分析判断是否为合适的建仓时机。"

            # 4. 获取用户偏好
            prefs_context = ""
            if self._checkpoint_manager:
                prefs_context = await self._checkpoint_manager.get_preferences_as_context(user_id, "stock")
                if prefs_context:
                    logger.info(f"📋 [{stock_code}] 已加载用户偏好")

            # 5. 发送给远端 Qwen 获取结构化建议
            logger.info(f"🌐 [{stock_code}] 调用远端 Qwen 获取操作建议...")
            advice = await self._get_qwen_advice(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                analysis_report=analysis_report,
                position_context=position_context,
                prefs_context=prefs_context,
                position_data=position_data,
            )

            if advice:
                logger.info(f"✅ [{stock_code}] 分析完成: {advice.action.value}")
                return SkillOutput(
                    success=True,
                    result=advice.to_dict()
                )
            else:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="获取操作建议失败"
                )

        except Exception as e:
            logger.error(f"❌ [{stock_code}] 监控分析失败: {e}")
            import traceback
            traceback.print_exc()
            return SkillOutput(
                success=False,
                result=None,
                error=str(e)
            )

    async def _get_stock_data(self, stock_code: str, days: int = 60) -> Optional[Dict[str, Any]]:
        """获取股票原始行情数据（OHLCV）

        使用公共模块从 AKShare 获取原始数据，不做技术指标计算，
        让 Finance LLM 自己分析和判断技术形态。
        """
        return await fetch_stock_raw_data(
            stock_code=stock_code,
            days=days,
            recent_days=60,  # 日线数据60天
            weekly_periods=20,  # 周线数据20周
            monthly_periods=12,  # 月线数据12月
        )

    async def _generate_analysis(self, stock_data: Dict[str, Any]) -> str:
        """使用本地 Finance LLM 分析原始行情数据

        使用公共模块构建 prompt，让 Finance LLM 自行计算和分析技术指标。
        """
        basic = stock_data.get("basic_info", {})
        quote = stock_data.get("latest_quote", {})
        recent_ohlcv = stock_data.get("recent_ohlcv", [])
        weekly_ohlcv = stock_data.get("weekly_ohlcv", [])
        monthly_ohlcv = stock_data.get("monthly_ohlcv", [])

        prompt = build_finance_llm_prompt(
            stock_name=basic.get("name", "N/A"),
            stock_code=basic.get("symbol", "N/A"),
            latest_quote=quote,
            recent_ohlcv=recent_ohlcv,
            analysis_type="brief",  # 盘中监控使用简要分析
            weekly_ohlcv=weekly_ohlcv,
            monthly_ohlcv=monthly_ohlcv,
        )

        try:
            llm = ChatOllama(
                model=self.FINANCE_MODEL,
                base_url=ollama_config.base_url,
                temperature=0.3,
            )
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Finance LLM 调用失败: {e}")
            return "Analysis unavailable"

    def _format_position_context(self, position: Dict[str, Any], current_price: float) -> str:
        """格式化持仓上下文"""
        shares = position.get("shares", 0)
        cost_price = position.get("cost_price", 0)
        cost_amount = position.get("cost_amount", 0)
        first_buy_date = position.get("first_buy_date", "未知")

        current_value = shares * current_price
        profit_loss = current_value - cost_amount
        profit_loss_pct = (profit_loss / cost_amount * 100) if cost_amount > 0 else 0

        return f"""## 用户持仓信息
- 持有股数: {shares:.0f} 股
- 成本价: ¥{cost_price:.2f}
- 投入成本: ¥{cost_amount:.2f}
- 当前市值: ¥{current_value:.2f}
- 浮动盈亏: ¥{profit_loss:.2f} ({profit_loss_pct:+.2f}%)
- 首次买入: {first_buy_date}
"""

    def _format_preferences_context(self, prefs: Dict[str, Any]) -> str:
        """格式化用户偏好上下文"""
        lines = ["## 用户投资偏好"]

        risk_map = {
            "conservative": "保守型（偏好低风险）",
            "moderate": "稳健型（适度风险）",
            "aggressive": "激进型（可接受高风险）",
        }
        horizon_map = {
            "short-term": "短线交易（几天到几周）",
            "medium-term": "中线波段（几周到几个月）",
            "long-term": "长线投资（数月到数年）",
        }

        if prefs.get("risk_tolerance"):
            lines.append(f"- 风险偏好: {risk_map.get(prefs['risk_tolerance'], prefs['risk_tolerance'])}")
        if prefs.get("analysis_horizon"):
            lines.append(f"- 投资周期: {horizon_map.get(prefs['analysis_horizon'], prefs['analysis_horizon'])}")
        if prefs.get("preferred_sectors"):
            lines.append(f"- 偏好板块: {', '.join(prefs['preferred_sectors'])}")

        return "\n".join(lines) if len(lines) > 1 else ""

    async def _get_qwen_advice(
        self,
        stock_code: str,
        stock_name: str,
        current_price: float,
        analysis_report: str,
        position_context: str,
        prefs_context: str,
        position_data: Optional[Dict[str, Any]],
    ) -> Optional[StockAdvice]:
        """调用远端 Qwen 获取结构化操作建议"""

        # 计算盈亏
        profit_loss_pct = None
        if position_data and position_data.get("cost_amount", 0) > 0:
            shares = position_data.get("shares", 0)
            cost_amount = position_data.get("cost_amount", 0)
            current_value = shares * current_price
            profit_loss_pct = (current_value - cost_amount) / cost_amount * 100

        # 从 md 文件加载 prompt 模板
        template = _load_prompt_template("qwen_trade_advice.md")
        if not template:
            logger.error("无法加载 qwen_trade_advice.md prompt 模板")
            return None

        prompt = template.format(
            stock_name=stock_name,
            stock_code=stock_code,
            current_price=current_price,
            analysis_report=analysis_report,
            position_context=position_context,
            prefs_context=prefs_context,
        )

        try:
            client = self._get_qwen_client()
            response = await client.chat.completions.create(
                model=qwen_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content

            # 解析 JSON
            # 尝试提取 JSON 块
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            result = json.loads(json_str)

            # 映射操作类型
            action_map = {
                "持有": TradeAction.HOLD,
                "买入": TradeAction.BUY,
                "卖出": TradeAction.SELL,
                "建仓": TradeAction.OPEN_POSITION,
            }
            action = action_map.get(result.get("action", "持有"), TradeAction.HOLD)

            return StockAdvice(
                stock_code=stock_code,
                stock_name=stock_name,
                current_price=current_price,
                action=action,
                reason=result.get("reason", ""),
                position_shares=position_data.get("shares") if position_data else None,
                cost_price=position_data.get("cost_price") if position_data else None,
                profit_loss_pct=profit_loss_pct,
                analysis_summary=result.get("analysis_summary", ""),
                llm_request=prompt,  # 保存发送给 LLM 的请求报文
            )

        except Exception as e:
            logger.error(f"Qwen 调用失败: {e}")
            return None

    async def format_email_content(self, advice: StockAdvice) -> str:
        """将操作建议格式化为邮件内容

        使用本地 Ollama Qwen 模型整理成自然语言
        """
        from langchain_ollama import ChatOllama

        prompt = f"""请将以下股票操作建议整理成一封简洁专业的中文提醒邮件。

股票信息：
- 股票名称: {advice.stock_name}
- 股票代码: {advice.stock_code}
- 当前价格: ¥{advice.current_price:.2f}

操作建议: {advice.action.value}
建议理由: {advice.reason}
技术分析: {advice.analysis_summary}

"""

        if advice.position_shares:
            prompt += f"""持仓情况：
- 持有股数: {advice.position_shares:.0f} 股
- 成本价: ¥{advice.cost_price:.2f}
- 当前盈亏: {advice.profit_loss_pct:+.2f}%

"""

        prompt += """请输出邮件正文内容（不需要标题和署名），要求：
1. 语言简洁专业
2. 突出关键信息（操作建议、理由）
3. 包含必要的风险提示
4. 使用 Markdown 格式"""

        try:
            llm = ChatOllama(
                model=ollama_config.model,  # 使用本地 qwen3.5:9b
                base_url=ollama_config.base_url,
                temperature=0.5,
            )
            response = await llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            # 备用：直接构建简单邮件
            logger.warning(f"本地模型调用失败，使用备用格式: {e}")
            return self._build_fallback_email(advice)

    def _build_fallback_email(self, advice: StockAdvice) -> str:
        """备用邮件格式"""
        emoji = {"买入": "🟢", "卖出": "🔴", "持有": "⏸️", "建仓": "🆕"}.get(advice.action.value, "📊")

        content = f"""## {emoji} 股票操作提醒

**{advice.stock_name}** ({advice.stock_code})

| 项目 | 内容 |
|------|------|
| 当前价格 | ¥{advice.current_price:.2f} |
| 操作建议 | **{advice.action.value}** |

### 📋 建议理由
{advice.reason}

### 📈 技术分析
{advice.analysis_summary}
"""

        if advice.position_shares:
            content += f"""
### 💼 持仓情况
| 项目 | 数值 |
|------|------|
| 持有股数 | {advice.position_shares:.0f} 股 |
| 成本价 | ¥{advice.cost_price:.2f} |
| 当前盈亏 | {advice.profit_loss_pct:+.2f}% |
"""

        content += """
---
*本提醒由 Jarvis 智能助手自动生成，仅供参考，不构成投资建议。*
"""
        return content
