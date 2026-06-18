"""股票报告生成 Skill - 翻译并增强分析报告

流程：
1. 接收 StockRetrieverSkill 返回的 analysis_report（已由 Finance LLM 生成）
2. 搜索股票最新新闻 (通过 MCP Tavily)
3. 使用远程 Qwen API 将英文报告翻译成中文，并注入新闻、偏好和持仓建议

支持用户偏好注入：根据用户偏好调整分析风格和建议
支持用户持仓注入：根据用户持仓给出个性化操作建议
支持 MCP 搜索：获取股票最新新闻和市场信息

使用 @dynamic_prompt 装饰器自动获取持仓和偏好上下文
"""
from typing import Any, Dict, Optional
import json
from pathlib import Path

from openai import AsyncOpenAI

from .base import BaseSkill, SkillOutput
from ..config import qwen_config
from ..storage.checkpoint import CheckpointManager
from ..utils.model_wrapper import (
    dynamic_prompt,
    PromptContext,
    build_prompt_sections,
)


# 持仓建议模板路径
POSITION_ADVICE_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "stock_position_advice.md"


class StockReportGeneratorSkill(BaseSkill):
    """股票报告生成 Skill - 翻译并增强 Finance LLM 分析报告

    接收 StockRetrieverSkill 返回的 analysis_report，翻译成中文并注入：
    - 用户偏好（调整分析风格和投资建议）
    - 用户持仓（给出个性化操作建议）
    - 最新新闻（来自 MCP Tavily 搜索）

    使用 @dynamic_prompt 装饰器自动获取上下文
    """

    name = "stock_report_generator"
    description = "将 Finance LLM 的英文分析报告翻译成中文，并注入用户偏好、持仓和最新新闻"

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        enable_mcp_search: bool = True,
    ):
        super().__init__()
        self._qwen_client: Optional[AsyncOpenAI] = None
        self._checkpoint_manager = checkpoint_manager
        self._enable_mcp_search = enable_mcp_search
        self._mcp_search_skill = None
        self._position_advice_template = self._load_position_advice_template()

    def set_checkpoint_manager(self, manager: CheckpointManager):
        """设置 CheckpointManager"""
        self._checkpoint_manager = manager

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

    async def _search_stock_news(self, stock_name: str, stock_code: str) -> str:
        """搜索股票最新新闻"""
        search_skill = await self._get_mcp_search_skill()
        if not search_skill:
            return ""

        try:
            print(f"🔍 [{stock_code}] 搜索最新新闻 (MCP Tavily)...")
            result = await search_skill.search_stock_news(stock_name, stock_code)

            if result.success and result.result:
                print(f"✅ [{stock_code}] 获取到新闻信息")
                return result.result
            else:
                print(f"⚠️ [{stock_code}] 新闻搜索无结果")
                return ""

        except Exception as e:
            print(f"⚠️ [{stock_code}] 新闻搜索失败: {e}")
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
                max_retries=qwen_config.max_retries,
            )
        return self._qwen_client

    def _load_position_advice_template(self) -> Optional[str]:
        """加载持仓建议模板"""
        try:
            with open(POSITION_ADVICE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"加载持仓建议模板失败: {e}")
            return None

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "stock_data_json": {
                    "type": "string",
                    "description": "由 stock_retriever 返回的 JSON 数据，包含 analysis_report 字段"
                },
                "user_id": {
                    "type": "string",
                    "description": "用户标识符，用于获取用户偏好",
                    "default": "default"
                }
            },
            "required": ["stock_data_json"]
        }

    async def execute(self, stock_data_json: str, user_id: str = "default", **kwargs) -> SkillOutput:
        """执行报告翻译和增强流程

        接收 StockRetrieverSkill 返回的数据（包含 analysis_report），
        翻译成中文并注入用户偏好、持仓和最新新闻。
        """
        try:
            # 1. 解析JSON数据
            try:
                data = json.loads(stock_data_json)
            except json.JSONDecodeError as e:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"JSON解析失败: {str(e)}"
                )

            # 2. 提取 Finance LLM 生成的英文分析报告
            english_report = data.get("analysis_report", "")
            if not english_report:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="数据中缺少 analysis_report 字段"
                )

            basic = data.get("basic_info", {})
            stock_code = basic.get("symbol", "")
            stock_name = basic.get("name", "")

            print(f"📝 [{stock_code}] 接收到 Finance LLM 报告 ({len(english_report)} 字符)")

            # 3. 搜索最新新闻 (MCP Tavily)
            news_context = await self._search_stock_news(stock_name, stock_code)

            # 4. 调用翻译方法（使用 @dynamic_prompt 自动注入持仓和偏好）
            chinese_report = await self.translate_to_chinese(
                english_report=english_report,
                data=data,
                news_context=news_context,
                user_id=user_id,
                stock_code=stock_code,
            )

            return SkillOutput(success=True, result=chinese_report)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return SkillOutput(
                success=False,
                result=None,
                error=f"报告生成失败: {str(e)}"
            )

    @dynamic_prompt(
        category="stock",
        include_preferences=True,
        include_positions=True,
        include_news=False,  # 新闻在 execute 中单独获取
        position_template_path=POSITION_ADVICE_TEMPLATE_PATH,
    )
    async def translate_to_chinese(
        self,
        english_report: str,
        data: Dict[str, Any],
        news_context: str = "",
        _prompt_context: Optional[PromptContext] = None,
        **kwargs
    ) -> str:
        """将英文报告翻译成中文（使用远程 Qwen API）

        使用 @dynamic_prompt 装饰器自动注入：
        - _prompt_context.preferences: 用户偏好
        - _prompt_context.positions: 用户持仓

        Args:
            english_report: 英文报告内容
            data: 股票数据字典
            news_context: 最新新闻上下文（从 execute 传入）
            _prompt_context: 自动注入的上下文（包含 preferences, positions）
        """
        basic = data.get("basic_info", {})
        quote = data.get("latest_quote", {})
        symbol = basic.get("symbol", "")
        name = basic.get("name", "")
        current_price = quote.get("close", 0)

        print(f"🔄 [{symbol}] 调用远程 Qwen API ({qwen_config.model}) 翻译成中文...")

        # 使用 @dynamic_prompt 注入的上下文
        ctx = _prompt_context or PromptContext()

        # 更新 context 中的价格信息（用于模板）
        ctx.current_price = current_price
        ctx.news = news_context

        # 构建 prompt 各部分
        sections = build_prompt_sections(ctx, self._position_advice_template)

        prompt = f"""你是一位专业的金融翻译专家。请将以下英文股票分析报告翻译成中文。

翻译要求：
1. 保持专业金融术语的准确性
2. 语言流畅自然，符合中文表达习惯
3. 保留原报告的结构和格式（标题、列表、表格等）
4. 关键结论和操作建议要**加粗**突出显示
5. 价格、百分比等数字保持不变
6. 股票名称使用: {name} ({symbol})
7. 使用适当的 emoji 增强可读性：
   - 📊 行情概览
   - 📈 趋势分析
   - 🎯 技术信号
   - ⚖️ 多空分析
   - ⚠️ 风险评估
   - 💡 操作建议
   - 💼 持仓操作建议
   - 📰 市场新闻
{sections.get('news_section', '')}{sections.get('position_section', '')}
英文报告：
---
{english_report}
---

请输出翻译后的中文报告（保持 Markdown 格式）：
"""

        try:
            client = self._get_qwen_client()
            response = await client.chat.completions.create(
                model=qwen_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=qwen_config.temperature,
            )
            result = response.choices[0].message.content

            # 检查是否返回空内容
            if not result or len(result.strip()) < 50:
                print(f"⚠️ [{symbol}] Qwen API 返回内容过短或为空，使用英文原文")
                return f"# {name} ({symbol}) 分析报告\n\n{english_report}"

            print(f"✅ [{symbol}] 中文翻译完成 ({len(result)} 字符)")
            return result

        except Exception as e:
            # 翻译失败时返回英文原文
            print(f"⚠️ [{symbol}] Qwen API 翻译失败: {e}")
            return f"[翻译失败，以下为英文原文]\n\n{english_report}"
