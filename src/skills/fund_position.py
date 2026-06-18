"""Fund Position Skills - 基金持仓管理

提供基金持仓的更新、查询和清仓功能，用于长期记忆存储。
"""
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillOutput
from ..storage.checkpoint import CheckpointManager


# 持仓操作识别规则
POSITION_OPERATION_PATTERNS = {
    "set": {
        # "设置持仓" - 直接设置当前持仓金额（不是增量，而是绝对值）
        # 例如: "我现在持有基金008089为27225元"、"008089持有5万"
        "keywords": ["持有", "目前持有", "现在持有", "当前持有"],
        "amount_patterns": [
            r"(?:持有|为)\s*(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(?:持有|为)\s*(\d+(?:\.\d+)?)\s*万",
            r"(?:持有|为)\s*(\d+(?:\.\d+)?)\s*千",
            r"(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
        ],
    },
    "buy": {
        "keywords": ["加仓", "买入", "申购", "定投", "买了", "投了", "加了", "购买", "申购了"],
        "amount_patterns": [
            r"(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
            r"(?:金额|投入|买了?|加了?)\s*[:：]?\s*(\d+(?:\.\d+)?)",
            r"了\s*(\d+(?:\.\d+)?)\s*(?:块|元|万|千)?",
        ],
    },
    "sell": {
        "keywords": ["卖出", "赎回", "减仓", "卖了", "赎了", "减了", "赎回了"],
        "amount_patterns": [
            r"(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
            r"(?:金额|卖了?|赎了?|减了?)\s*[:：]?\s*(\d+(?:\.\d+)?)",
        ],
    },
    "clear": {
        "keywords": ["清仓", "全部卖出", "全部赎回", "清空"],
    },
    "query": {
        # 注意: "持有" 已移到 "set" 操作，因为带金额时是设置持仓
        # 这里只保留纯查询关键词
        "keywords": ["持仓", "仓位", "买了多少", "投了多少", "有多少", "仓位情况", "持仓情况", "持仓查询", "查看持仓"],
    },
}


def detect_position_operation(text: str) -> Dict[str, Any]:
    """检测用户输入中的持仓操作意图

    Args:
        text: 用户输入文本

    Returns:
        {
            "operation": "set" | "buy" | "sell" | "clear" | "query" | None,
            "fund_code": str | None,
            "amount": float | None,
            "matched_keyword": str | None,
        }
    """
    result = {
        "operation": None,
        "fund_code": None,
        "amount": None,
        "matched_keyword": None,
    }

    # 提取基金代码（6位数字）
    # 使用更宽松的模式，不依赖 \b word boundary（在中文中可能不工作）
    code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if code_match:
        result["fund_code"] = code_match.group(1)

    # 先尝试提取金额（用于判断 "持有" 是设置还是查询）
    def extract_amount(text: str, patterns: list) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1)
                amount = float(amount_str)

                # 处理单位
                context = text[max(0, match.start()-5):min(len(text), match.end()+5)]
                if "万" in context:
                    if amount < 1000:  # 可能是 "5万" 这种情况
                        amount *= 10000
                elif "千" in context:
                    if amount < 10000:
                        amount *= 1000

                return amount
        return None

    # 检测操作类型（按优先级顺序）
    # 优先级: set > buy > sell > clear > query
    priority_order = ["set", "buy", "sell", "clear", "query"]

    for operation in priority_order:
        if operation not in POSITION_OPERATION_PATTERNS:
            continue

        config = POSITION_OPERATION_PATTERNS[operation]
        keywords = config.get("keywords", [])

        for keyword in keywords:
            if keyword in text:
                # 对于 "set" 操作，需要同时有金额才算设置持仓
                # 否则可能只是询问 "我持有什么"
                if operation == "set":
                    amount_patterns = config.get("amount_patterns", [])
                    amount = extract_amount(text, amount_patterns)
                    if amount is not None:
                        result["operation"] = "set"
                        result["matched_keyword"] = keyword
                        result["amount"] = amount
                        return result
                    # 没有金额，跳过 set，继续检查其他操作
                    continue

                result["operation"] = operation
                result["matched_keyword"] = keyword
                break

        if result["operation"]:
            break

    # 如果是买入或卖出操作，尝试提取金额
    if result["operation"] in ("buy", "sell"):
        amount_patterns = POSITION_OPERATION_PATTERNS[result["operation"]].get("amount_patterns", [])
        result["amount"] = extract_amount(text, amount_patterns)

    # 特殊处理：如果没有匹配任何操作，但文本包含 "持有" 且有金额，视为 set
    if result["operation"] is None and "持有" in text:
        set_config = POSITION_OPERATION_PATTERNS.get("set", {})
        amount_patterns = set_config.get("amount_patterns", [])
        amount = extract_amount(text, amount_patterns)
        if amount is not None:
            result["operation"] = "set"
            result["matched_keyword"] = "持有"
            result["amount"] = amount

    return result


class UpdateFundPositionSkill(BaseSkill):
    """更新基金持仓

    支持买入、卖出、清仓操作，自动计算成本净值。
    """

    name = "update_fund_position"
    description = "更新用户基金持仓（设置/买入/卖出/清仓），自动计算成本净值"

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户标识符",
                },
                "fund_code": {
                    "type": "string",
                    "description": "基金代码（6位数字）",
                },
                "operation": {
                    "type": "string",
                    "description": "操作类型: set（设置持仓）, buy（买入）, sell（卖出）, clear（清仓）",
                    "enum": ["set", "buy", "sell", "clear"],
                },
                "amount": {
                    "type": "number",
                    "description": "交易金额（买入为正，卖出为负）",
                },
                "shares": {
                    "type": "number",
                    "description": "交易份额（可选，如果提供则优先使用）",
                },
                "nav": {
                    "type": "number",
                    "description": "当前净值（用于计算份额，可选）",
                },
                "fund_name": {
                    "type": "string",
                    "description": "基金名称（可选）",
                },
            },
            "required": ["user_id", "fund_code", "operation"],
        }

    async def execute(
        self,
        user_id: str,
        fund_code: str,
        operation: str,
        amount: Optional[float] = None,
        shares: Optional[float] = None,
        nav: Optional[float] = None,
        fund_name: Optional[str] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs,
    ) -> SkillOutput:
        """执行持仓更新

        Args:
            user_id: 用户标识符
            fund_code: 基金代码
            operation: 操作类型 (set/buy/sell/clear)
            amount: 交易金额
            shares: 交易份额
            nav: 当前净值
            fund_name: 基金名称
            checkpoint_manager: CheckpointManager 实例

        Returns:
            SkillOutput 包含更新结果
        """
        try:
            manager = checkpoint_manager or self._checkpoint_manager
            if not manager:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="CheckpointManager 未配置",
                )

            # 验证操作类型
            valid_operations = ["set", "buy", "sell", "clear"]
            if operation not in valid_operations:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"无效的操作类型: {operation}，有效类型: {valid_operations}",
                )

            # 对于设置/买入/卖出，需要金额或份额
            if operation in ("set", "buy", "sell") and amount is None and shares is None:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"{operation} 操作需要提供交易金额或份额",
                )

            # 计算份额变化和金额变化
            shares_delta = 0.0
            amount_delta = 0.0

            if operation == "clear":
                # 清仓操作
                pass
            elif operation == "set":
                # 设置持仓 - 直接设置绝对值，而不是增量
                if shares:
                    shares_delta = abs(shares)
                if amount:
                    amount_delta = abs(amount)
            elif operation == "buy":
                if shares:
                    shares_delta = abs(shares)
                if amount:
                    amount_delta = abs(amount)
            elif operation == "sell":
                if shares:
                    shares_delta = -abs(shares)
                if amount:
                    amount_delta = -abs(amount)

            # 执行更新
            result = await manager.save_fund_position(
                user_id=user_id,
                fund_code=fund_code,
                fund_name=fund_name,
                shares_delta=shares_delta,
                amount_delta=amount_delta,
                nav=nav,
                operation=operation,
            )

            # 构建响应消息
            if operation == "clear":
                message = f"基金 {fund_code} 已清仓"
            elif operation == "set":
                fund_display = fund_name or fund_code
                message = f"✅ 基金 {fund_display} ({fund_code}) 持仓已记录"
                if amount:
                    message += f"\n   💰 持仓金额: ¥{amount:.2f}"
                if result.get("new_shares") and result["new_shares"] > 0:
                    message += f"\n   📊 持有份额: {result['new_shares']:.4f} 份"
                if result.get("new_cost_nav") and result["new_cost_nav"] > 0:
                    message += f"\n   📈 成本净值: {result['new_cost_nav']:.4f}"
                elif amount and (not result.get("new_shares") or result["new_shares"] == 0):
                    message += f"\n   ⚠️ 未能获取当前净值，仅记录金额。后续可通过加仓操作补充净值信息。"
            elif operation == "buy":
                message = f"基金 {fund_code} 买入记录已保存"
                if amount:
                    message += f"，金额 ¥{amount:.2f}"
                if result.get("new_shares"):
                    message += f"，当前持有 {result['new_shares']:.4f} 份"
                if result.get("new_cost_nav"):
                    message += f"，成本净值 {result['new_cost_nav']:.4f}"
            else:  # sell
                message = f"基金 {fund_code} 卖出记录已保存"
                if amount:
                    message += f"，金额 ¥{abs(amount):.2f}"
                if result.get("new_shares"):
                    message += f"，剩余 {result['new_shares']:.4f} 份"

            result["message"] = message

            return SkillOutput(
                success=True,
                result=result,
            )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"更新持仓失败: {str(e)}",
            )


class GetFundPositionsSkill(BaseSkill):
    """查询基金持仓

    支持查询单个基金或用户所有持仓。
    """

    name = "get_fund_positions"
    description = "查询用户基金持仓，支持查询单个基金或所有持仓"

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户标识符",
                },
                "fund_code": {
                    "type": "string",
                    "description": "基金代码（可选，不指定则返回所有持仓）",
                },
                "as_context": {
                    "type": "boolean",
                    "description": "是否以 LLM 上下文格式返回，默认 False",
                },
            },
            "required": ["user_id"],
        }

    async def execute(
        self,
        user_id: str,
        fund_code: Optional[str] = None,
        as_context: bool = False,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs,
    ) -> SkillOutput:
        """查询持仓信息

        Args:
            user_id: 用户标识符
            fund_code: 基金代码（可选）
            as_context: 是否以 LLM 上下文格式返回
            checkpoint_manager: CheckpointManager 实例

        Returns:
            SkillOutput 包含持仓信息
        """
        try:
            manager = checkpoint_manager or self._checkpoint_manager
            if not manager:
                return SkillOutput(
                    success=False,
                    result=None,
                    error="CheckpointManager 未配置",
                )

            if as_context:
                # 返回格式化的上下文字符串
                context = await manager.get_positions_as_context(user_id, fund_code)
                return SkillOutput(
                    success=True,
                    result=context,
                )

            if fund_code:
                # 查询单个基金
                position = await manager.get_fund_position(user_id, fund_code)
                if position:
                    return SkillOutput(
                        success=True,
                        result={
                            "user_id": user_id,
                            "position": position,
                            "message": self._format_position_message(position),
                        },
                    )
                else:
                    return SkillOutput(
                        success=True,
                        result={
                            "user_id": user_id,
                            "position": None,
                            "message": f"未找到基金 {fund_code} 的持仓记录",
                        },
                    )
            else:
                # 查询所有持仓
                positions = await manager.get_all_fund_positions(user_id)
                return SkillOutput(
                    success=True,
                    result={
                        "user_id": user_id,
                        "positions": positions,
                        "count": len(positions),
                        "message": self._format_positions_summary(positions),
                    },
                )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"查询持仓失败: {str(e)}",
            )

    def _format_position_message(self, position: Dict[str, Any]) -> str:
        """格式化单个持仓信息"""
        fund_name = position.get("fund_name") or position["fund_code"]
        shares = position.get("shares", 0)
        cost_amount = position.get("cost_amount", 0)
        cost_nav = position.get("cost_nav")

        lines = [
            f"📊 **{fund_name}** ({position['fund_code']})",
            f"  - 持有份额: {shares:.4f} 份",
            f"  - 累计投入: ¥{cost_amount:.2f}",
        ]

        if cost_nav:
            lines.append(f"  - 成本净值: {cost_nav:.4f}")

        if position.get("first_buy_date"):
            lines.append(f"  - 首次买入: {position['first_buy_date']}")

        if position.get("last_trade_date"):
            lines.append(f"  - 最后交易: {position['last_trade_date']}")

        return "\n".join(lines)

    def _format_positions_summary(self, positions: List[Dict[str, Any]]) -> str:
        """格式化持仓汇总信息"""
        if not positions:
            return "📋 您当前没有基金持仓记录"

        total_cost = sum(p.get("cost_amount", 0) for p in positions)

        lines = [
            f"📋 **基金持仓汇总**",
            f"  - 持仓数量: {len(positions)} 只基金",
            f"  - 累计投入: ¥{total_cost:.2f}",
            "",
            "**持仓明细**:",
        ]

        for i, pos in enumerate(positions, 1):
            fund_name = pos.get("fund_name") or pos["fund_code"]
            shares = pos.get("shares", 0)
            cost_amount = pos.get("cost_amount", 0)
            cost_nav = pos.get("cost_nav")

            detail = f"{i}. **{fund_name}** ({pos['fund_code']})"
            detail += f" - {shares:.2f}份, ¥{cost_amount:.2f}"
            if cost_nav:
                detail += f", 成本{cost_nav:.4f}"

            lines.append(detail)

        return "\n".join(lines)
