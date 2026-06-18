"""Stock Position Skills - 股票持仓管理

提供股票持仓的更新、查询和清仓功能，用于长期记忆存储。
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
        # 例如: "我现在持有股票600588为27225元"、"600588持有5万"
        "keywords": ["持有", "目前持有", "现在持有", "当前持有"],
        "amount_patterns": [
            r"(?:持有|为)\s*(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(?:持有|为)\s*(\d+(?:\.\d+)?)\s*万",
            r"(?:持有|为)\s*(\d+(?:\.\d+)?)\s*千",
            r"(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
        ],
        "shares_patterns": [
            r"(\d+)\s*股",
            r"(\d+)\s*手",  # 1手=100股
        ],
    },
    "buy": {
        "keywords": ["加仓", "买入", "建仓", "买了", "投了", "加了", "购买"],
        "amount_patterns": [
            r"(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
            r"(?:金额|投入|买了?|加了?)\s*[:：]?\s*(\d+(?:\.\d+)?)",
            r"了\s*(\d+(?:\.\d+)?)\s*(?:块|元|万|千)?",
        ],
        "shares_patterns": [
            r"(\d+)\s*股",
            r"(\d+)\s*手",
        ],
    },
    "sell": {
        "keywords": ["卖出", "减仓", "卖了", "减了", "抛出", "出货"],
        "amount_patterns": [
            r"(\d+(?:\.\d+)?)\s*(?:块|元)",
            r"(\d+(?:\.\d+)?)\s*万",
            r"(\d+(?:\.\d+)?)\s*千",
            r"(?:金额|卖了?|减了?)\s*[:：]?\s*(\d+(?:\.\d+)?)",
        ],
        "shares_patterns": [
            r"(\d+)\s*股",
            r"(\d+)\s*手",
        ],
    },
    "clear": {
        "keywords": ["清仓", "全部卖出", "全部卖掉", "清空"],
    },
    "query": {
        # 注意: "持有" 已移到 "set" 操作，因为带金额时是设置持仓
        # 这里只保留纯查询关键词
        "keywords": ["持仓", "仓位", "买了多少", "投了多少", "有多少", "仓位情况", "持仓情况", "持仓查询", "查看持仓"],
    },
}


def detect_stock_position_operation(text: str) -> Dict[str, Any]:
    """检测用户输入中的股票持仓操作意图

    Args:
        text: 用户输入文本

    Returns:
        {
            "operation": "set" | "buy" | "sell" | "clear" | "query" | None,
            "stock_code": str | None,
            "amount": float | None,
            "shares": int | None,
            "matched_keyword": str | None,
        }
    """
    result = {
        "operation": None,
        "stock_code": None,
        "amount": None,
        "shares": None,
        "matched_keyword": None,
    }

    # 提取股票代码（6位数字）
    code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if code_match:
        result["stock_code"] = code_match.group(1)

    # 提取金额
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

    # 提取股数
    def extract_shares(text: str, patterns: list) -> Optional[int]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                shares = int(match.group(1))
                # 如果是"手"，转换为股（1手=100股）
                if "手" in text[max(0, match.start()-2):min(len(text), match.end()+2)]:
                    shares *= 100
                return shares
        return None

    # 检测操作类型（按优先级顺序）
    priority_order = ["set", "buy", "sell", "clear", "query"]

    for operation in priority_order:
        if operation not in POSITION_OPERATION_PATTERNS:
            continue

        config = POSITION_OPERATION_PATTERNS[operation]
        keywords = config.get("keywords", [])

        for keyword in keywords:
            if keyword in text:
                # 对于 "set" 操作，需要同时有金额或股数才算设置持仓
                if operation == "set":
                    amount_patterns = config.get("amount_patterns", [])
                    shares_patterns = config.get("shares_patterns", [])
                    amount = extract_amount(text, amount_patterns)
                    shares = extract_shares(text, shares_patterns)
                    if amount is not None or shares is not None:
                        result["operation"] = "set"
                        result["matched_keyword"] = keyword
                        result["amount"] = amount
                        result["shares"] = shares
                        return result
                    # 没有金额或股数，跳过 set，继续检查其他操作
                    continue

                result["operation"] = operation
                result["matched_keyword"] = keyword
                break

        if result["operation"]:
            break

    # 如果是买入或卖出操作，尝试提取金额和股数
    if result["operation"] in ("buy", "sell"):
        config = POSITION_OPERATION_PATTERNS[result["operation"]]
        amount_patterns = config.get("amount_patterns", [])
        shares_patterns = config.get("shares_patterns", [])
        result["amount"] = extract_amount(text, amount_patterns)
        result["shares"] = extract_shares(text, shares_patterns)

    # 特殊处理：如果没有匹配任何操作，但文本包含 "持有" 且有金额/股数，视为 set
    if result["operation"] is None and "持有" in text:
        set_config = POSITION_OPERATION_PATTERNS.get("set", {})
        amount_patterns = set_config.get("amount_patterns", [])
        shares_patterns = set_config.get("shares_patterns", [])
        amount = extract_amount(text, amount_patterns)
        shares = extract_shares(text, shares_patterns)
        if amount is not None or shares is not None:
            result["operation"] = "set"
            result["matched_keyword"] = "持有"
            result["amount"] = amount
            result["shares"] = shares

    return result


class UpdateStockPositionSkill(BaseSkill):
    """更新股票持仓

    支持买入、卖出、清仓操作，自动计算成本价。
    当只提供金额或股数时，会自动获取当前股价来补全另一个值。
    """

    name = "update_stock_position"
    description = "更新用户股票持仓（设置/买入/卖出/清仓），自动计算成本价"

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    async def _get_current_price(self, stock_code: str) -> tuple[Optional[float], Optional[str]]:
        """获取股票当前价格和名称

        Args:
            stock_code: 股票代码

        Returns:
            (当前价格, 股票名称) 或 (None, None)
        """
        try:
            import akshare as ak

            # 判断市场
            if stock_code.startswith("6"):
                market = "sh"
            elif stock_code.startswith(("0", "3")):
                market = "sz"
            else:
                market = "bj"

            # 获取实时行情
            sina_symbol = f"{market}{stock_code}"
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                row = df[df["代码"] == stock_code]
                if not row.empty:
                    price = float(row.iloc[0]["最新价"])
                    name = str(row.iloc[0]["名称"])
                    return price, name

            # 备用：从日线数据获取最新收盘价
            df = ak.stock_zh_a_daily(symbol=sina_symbol, adjust="qfq")
            if df is not None and not df.empty:
                price = float(df.iloc[-1]["close"])
                # 获取股票名称
                stock_list = ak.stock_info_a_code_name()
                name = None
                if stock_list is not None:
                    name_row = stock_list[stock_list["code"] == stock_code]
                    if not name_row.empty:
                        name = str(name_row.iloc[0]["name"])
                return price, name

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"获取股票 {stock_code} 当前价格失败: {e}")

        return None, None

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "用户标识符",
                },
                "stock_code": {
                    "type": "string",
                    "description": "股票代码（6位数字）",
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
                    "description": "交易股数（可选，如果提供则优先使用）",
                },
                "price": {
                    "type": "number",
                    "description": "当前价格（用于计算股数，可选）",
                },
                "stock_name": {
                    "type": "string",
                    "description": "股票名称（可选）",
                },
            },
            "required": ["user_id", "stock_code", "operation"],
        }

    async def execute(
        self,
        user_id: str,
        stock_code: str,
        operation: str,
        amount: Optional[float] = None,
        shares: Optional[float] = None,
        price: Optional[float] = None,
        stock_name: Optional[str] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs,
    ) -> SkillOutput:
        """执行持仓更新

        Args:
            user_id: 用户标识符
            stock_code: 股票代码
            operation: 操作类型 (set/buy/sell/clear)
            amount: 交易金额
            shares: 交易股数
            price: 当前价格
            stock_name: 股票名称
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

            # 对于设置/买入/卖出，需要金额或股数
            if operation in ("set", "buy", "sell") and amount is None and shares is None:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"{operation} 操作需要提供交易金额或股数",
                )

            # 如果没有提供价格，且只提供了股数（没有金额），需要提示用户提供价格
            # 注意：只有金额没有股数的情况可以接受（直接记录金额）
            if price is None and operation in ("set", "buy", "sell"):
                if shares is not None and amount is None:
                    # 有股数无金额无价格：需要用户提供买入价格
                    return SkillOutput(
                        success=False,
                        result=None,
                        error=f"请提供买入单价，以便计算持仓金额（股数: {shares}股）",
                    )
                elif amount is not None and shares is None:
                    # 有金额无股数：尝试获取当前股价来计算股数（可选）
                    current_price, fetched_name = await self._get_current_price(stock_code)
                    if current_price:
                        price = current_price
                        if not stock_name and fetched_name:
                            stock_name = fetched_name

            # 根据价格补全金额或股数
            if price and price > 0:
                if amount is not None and shares is None:
                    # 有金额无股数：计算股数（按100股整数）
                    shares = int(amount / price / 100) * 100
                    if shares == 0 and amount >= price:
                        shares = int(amount / price)  # 允许零散股
                elif shares is not None and amount is None:
                    # 有股数无金额：计算金额
                    amount = shares * price

            # 计算股数变化和金额变化
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
            result = await manager.save_stock_position(
                user_id=user_id,
                stock_code=stock_code,
                stock_name=stock_name,
                shares_delta=shares_delta,
                amount_delta=amount_delta,
                price=price,
                operation=operation,
            )

            # 构建响应消息
            if operation == "clear":
                message = f"股票 {stock_code} 已清仓"
            elif operation == "set":
                stock_display = stock_name or stock_code
                message = f"✅ 股票 {stock_display} ({stock_code}) 持仓已记录"
                if amount:
                    message += f"\n   💰 持仓金额: ¥{amount:.2f}"
                if shares and shares > 0:
                    message += f"\n   📊 持有股数: {shares:.0f} 股"
                if price and price > 0:
                    message += f"\n   📈 当前股价: ¥{price:.2f}"
                if result.get("new_cost_price") and result["new_cost_price"] > 0:
                    message += f"\n   💵 成本价: ¥{result['new_cost_price']:.2f}"
            elif operation == "buy":
                stock_display = stock_name or stock_code
                message = f"✅ 股票 {stock_display} ({stock_code}) 买入记录已保存"
                if amount:
                    message += f"\n   💰 买入金额: ¥{amount:.2f}"
                if shares and shares > 0:
                    message += f"\n   📊 买入股数: {shares:.0f} 股"
                if price and price > 0:
                    message += f"\n   📈 买入价格: ¥{price:.2f}"
                if result.get("new_shares"):
                    message += f"\n   📦 当前持有: {result['new_shares']:.0f} 股"
                if result.get("new_cost_price"):
                    message += f"\n   💵 成本价: ¥{result['new_cost_price']:.2f}"
            else:  # sell
                stock_display = stock_name or stock_code
                message = f"✅ 股票 {stock_display} ({stock_code}) 卖出记录已保存"
                if amount:
                    message += f"\n   💰 卖出金额: ¥{abs(amount):.2f}"
                if shares and shares > 0:
                    message += f"\n   📊 卖出股数: {abs(shares):.0f} 股"
                if result.get("new_shares") is not None:
                    message += f"\n   📦 剩余持有: {result['new_shares']:.0f} 股"

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


class GetStockPositionsSkill(BaseSkill):
    """查询股票持仓

    支持查询单个股票或用户所有持仓。
    """

    name = "get_stock_positions"
    description = "查询用户股票持仓，支持查询单个股票或所有持仓"

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
                "stock_code": {
                    "type": "string",
                    "description": "股票代码（可选，不指定则返回所有持仓）",
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
        stock_code: Optional[str] = None,
        as_context: bool = False,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs,
    ) -> SkillOutput:
        """查询持仓信息

        Args:
            user_id: 用户标识符
            stock_code: 股票代码（可选）
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
                context = await manager.get_stock_positions_as_context(user_id, stock_code)
                return SkillOutput(
                    success=True,
                    result=context,
                )

            if stock_code:
                # 查询单个股票
                position = await manager.get_stock_position(user_id, stock_code)
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
                            "message": f"未找到股票 {stock_code} 的持仓记录",
                        },
                    )
            else:
                # 查询所有持仓
                positions = await manager.get_all_stock_positions(user_id)
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
        stock_name = position.get("stock_name") or position["stock_code"]
        shares = position.get("shares", 0)
        cost_amount = position.get("cost_amount", 0)
        cost_price = position.get("cost_price")

        lines = [
            f"📊 **{stock_name}** ({position['stock_code']})",
            f"  - 持有股数: {shares:.0f} 股",
            f"  - 累计投入: ¥{cost_amount:.2f}",
        ]

        if cost_price:
            lines.append(f"  - 成本价: ¥{cost_price:.2f}")

        if position.get("first_buy_date"):
            lines.append(f"  - 首次买入: {position['first_buy_date']}")

        if position.get("last_trade_date"):
            lines.append(f"  - 最后交易: {position['last_trade_date']}")

        return "\n".join(lines)

    def _format_positions_summary(self, positions: List[Dict[str, Any]]) -> str:
        """格式化持仓汇总信息"""
        if not positions:
            return "📋 您当前没有股票持仓记录"

        total_cost = sum(p.get("cost_amount", 0) for p in positions)

        lines = [
            f"📋 **股票持仓汇总**",
            f"  - 持仓数量: {len(positions)} 只股票",
            f"  - 累计投入: ¥{total_cost:.2f}",
            "",
            "**持仓明细**:",
        ]

        for i, pos in enumerate(positions, 1):
            stock_name = pos.get("stock_name") or pos["stock_code"]
            shares = pos.get("shares", 0)
            cost_amount = pos.get("cost_amount", 0)
            cost_price = pos.get("cost_price")

            detail = f"{i}. **{stock_name}** ({pos['stock_code']})"
            detail += f" - {shares:.0f}股, ¥{cost_amount:.2f}"
            if cost_price:
                detail += f", 成本¥{cost_price:.2f}"

            lines.append(detail)

        return "\n".join(lines)
