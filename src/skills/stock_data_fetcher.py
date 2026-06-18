"""股票原始数据获取模块 - 供 StockRetrieverSkill 和 StockMonitorSkill 共用

只负责从 AKShare 获取原始 OHLCV 数据（日线、周线、月线），不做技术指标计算。
技术分析由 Finance LLM 完成。
"""
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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


def _get_stock_name(stock_code: str) -> str:
    """获取股票名称"""
    import akshare as ak

    try:
        stock_list = ak.stock_info_a_code_name()
        if stock_list is not None and not stock_list.empty:
            name_row = stock_list[stock_list['code'] == stock_code]
            if not name_row.empty:
                name = name_row.iloc[0]['name']
                logger.info(f"[股票名称] {stock_code} -> {name}")
                return name
    except Exception as e:
        logger.warning(f"获取股票名称失败: {e}")

    return stock_code


def _fetch_realtime_quote(stock_code: str) -> Optional[Dict[str, Any]]:
    """获取股票实时行情

    尝试多个数据源获取当前价格、涨跌幅等数据。
    优先使用新浪接口（更稳定），备用东方财富接口。

    Args:
        stock_code: 股票代码（6位数字）

    Returns:
        实时行情字典，包含 price, open, high, low, volume, pct_change 等字段
        如果获取失败返回 None
    """
    import akshare as ak

    # 构建带市场前缀的代码（新浪接口需要）
    if stock_code.startswith("6"):
        sina_code = f"sh{stock_code}"
    elif stock_code.startswith(("0", "3")):
        sina_code = f"sz{stock_code}"
    else:
        sina_code = f"bj{stock_code}"

    # 优先使用新浪全量实时行情（更稳定）
    try:
        df = ak.stock_zh_a_spot()
        if df is not None and not df.empty:
            # 查找指定股票（新浪代码带市场前缀）
            code_col = 'code' if 'code' in df.columns else '代码'
            row = df[df[code_col] == sina_code]
            if not row.empty:
                row = row.iloc[0]
                # 自动检测列名（可能是中文或英文）
                def get_val(r, keys, default=None):
                    for k in keys:
                        if k in r.index and pd.notna(r[k]):
                            return r[k]
                    return default

                price = get_val(row, ['trade', '最新价', 'price'])
                quote = {
                    "price": float(price) if price else None,
                    "open": float(get_val(row, ['open', '今开'], 0) or 0),
                    "high": float(get_val(row, ['high', '最高'], 0) or 0),
                    "low": float(get_val(row, ['low', '最低'], 0) or 0),
                    "volume": int(float(get_val(row, ['volume', '成交量'], 0) or 0)),
                    "amount": float(get_val(row, ['amount', '成交额'], 0) or 0),
                    "pct_change": float(get_val(row, ['changepercent', '涨跌幅'], 0) or 0),
                    "name": str(get_val(row, ['name', '名称'], stock_code)),
                }
                if quote["price"]:
                    logger.info(f"[实时行情-新浪] {stock_code} 最新价: {quote['price']}, 涨跌幅: {quote['pct_change']}%")
                    return quote
    except Exception as e:
        logger.warning(f"获取实时行情失败 (stock_zh_a_spot): {e}")

    # 备用方案：东方财富 A 股实时行情
    try:
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            row = df[df['代码'] == stock_code]
            if not row.empty:
                row = row.iloc[0]
                quote = {
                    "price": float(row['最新价']) if pd.notna(row['最新价']) else None,
                    "open": float(row['今开']) if pd.notna(row['今开']) else None,
                    "high": float(row['最高']) if pd.notna(row['最高']) else None,
                    "low": float(row['最低']) if pd.notna(row['最低']) else None,
                    "volume": int(row['成交量']) if pd.notna(row['成交量']) else 0,
                    "amount": float(row['成交额']) if pd.notna(row['成交额']) else 0,
                    "pct_change": float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
                    "change": float(row['涨跌额']) if pd.notna(row['涨跌额']) else 0,
                    "turnover": float(row['换手率']) if pd.notna(row['换手率']) else 0,
                    "name": str(row['名称']) if pd.notna(row['名称']) else stock_code,
                }
                if quote["price"]:
                    logger.info(f"[实时行情-东财] {stock_code} 最新价: {quote['price']}, 涨跌幅: {quote['pct_change']}%")
                    return quote
    except Exception as e:
        logger.warning(f"获取实时行情失败 (stock_zh_a_spot_em): {e}")

    return None


def _fetch_daily_data(
    stock_code: str,
    market: str,
    start_date: str,
    end_date: str,
    days: int,
) -> Optional[pd.DataFrame]:
    """获取日线数据"""
    import akshare as ak

    df = None

    # 方法1: 新浪数据源（优先）
    try:
        sina_symbol = f"{market}{stock_code}"
        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if df is not None and not df.empty:
            df = df.rename(columns={
                "date": "date", "open": "open", "close": "close",
                "high": "high", "low": "low", "volume": "volume"
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            logger.info(f"[新浪日线] 获取 {stock_code} {len(df)} 条数据")
            return df
    except Exception as e:
        logger.warning(f"新浪日线数据源失败: {e}")

    # 方法2: 东方财富（备用）
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if df is not None and not df.empty:
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "涨跌幅": "pct_change"
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            logger.info(f"[东方财富日线] 获取 {stock_code} {len(df)} 条数据")
            return df
    except Exception as e:
        logger.warning(f"东方财富日线数据源失败: {e}")

    return None


def _fetch_weekly_data(
    stock_code: str,
    market: str,
    weeks: int = 20,
) -> Optional[pd.DataFrame]:
    """获取周线数据（使用新浪数据源）"""
    import akshare as ak
    import time

    # 计算日期范围
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=weeks * 7 + 60)).strftime("%Y%m%d")

    # 方法1: 新浪周线数据（优先）
    try:
        sina_symbol = f"{market}{stock_code}"
        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if df is not None and not df.empty:
            # 将日线数据转换为周线数据
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            # 按周重采样
            weekly_df = df.resample("W").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

            weekly_df = weekly_df.reset_index()
            weekly_df["pct_change"] = weekly_df["close"].pct_change() * 100
            weekly_df = weekly_df.tail(weeks).reset_index(drop=True)
            logger.info(f"[新浪周线] 获取 {stock_code} {len(weekly_df)} 条数据")
            return weekly_df
    except Exception as e:
        logger.warning(f"新浪周线数据获取失败: {e}")

    # 方法2: 东方财富（备用）
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="weekly",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "涨跌幅": "pct_change"
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").tail(weeks).reset_index(drop=True)
                logger.info(f"[东方财富周线] 获取 {stock_code} {len(df)} 条数据")
                return df
        except Exception as e:
            logger.warning(f"东方财富周线数据获取失败 (尝试 {attempt + 1}/2): {e}")
            if attempt < 1:
                time.sleep(1)

    return None


def _fetch_monthly_data(
    stock_code: str,
    market: str,
    months: int = 12,
) -> Optional[pd.DataFrame]:
    """获取月线数据（使用新浪数据源）"""
    import akshare as ak
    import time

    # 计算日期范围
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=months * 35 + 60)).strftime("%Y%m%d")

    # 方法1: 新浪数据源 - 从日线转换为月线（优先）
    try:
        sina_symbol = f"{market}{stock_code}"
        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if df is not None and not df.empty:
            # 将日线数据转换为月线数据
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            # 按月重采样
            monthly_df = df.resample("ME").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

            monthly_df = monthly_df.reset_index()
            monthly_df["pct_change"] = monthly_df["close"].pct_change() * 100
            monthly_df = monthly_df.tail(months).reset_index(drop=True)
            logger.info(f"[新浪月线] 获取 {stock_code} {len(monthly_df)} 条数据")
            return monthly_df
    except Exception as e:
        logger.warning(f"新浪月线数据获取失败: {e}")

    # 方法2: 东方财富（备用）
    for attempt in range(2):
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="monthly",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                    "涨跌幅": "pct_change"
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").tail(months).reset_index(drop=True)
                logger.info(f"[东方财富月线] 获取 {stock_code} {len(df)} 条数据")
                return df
        except Exception as e:
            logger.warning(f"东方财富月线数据获取失败 (尝试 {attempt + 1}/2): {e}")
            if attempt < 1:
                time.sleep(1)

    return None


def _df_to_ohlcv_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """将 DataFrame 转换为 OHLCV 字典列表"""
    if df is None or df.empty:
        return []

    # 确保有涨跌幅
    if "pct_change" not in df.columns:
        df = df.copy()
        df["pct_change"] = df["close"].pct_change() * 100

    result = df[["date", "open", "high", "low", "close", "volume", "pct_change"]].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    records = result.to_dict(orient="records")
    # 处理 NaN
    for r in records:
        if pd.isna(r.get("pct_change")):
            r["pct_change"] = 0.0
    return records


async def fetch_stock_raw_data(
    stock_code: str,
    days: int = 60,
    recent_days: int = 60,
    include_weekly: bool = True,
    include_monthly: bool = True,
    weekly_periods: int = 20,
    monthly_periods: int = 12,
) -> Optional[Dict[str, Any]]:
    """获取股票原始行情数据（日线 + 周线 + 月线）

    Args:
        stock_code: 股票代码（6位数字）
        days: 获取日线历史数据天数
        recent_days: 返回最近N天的日线数据供LLM分析（默认60天）
        include_weekly: 是否包含周线数据
        include_monthly: 是否包含月线数据
        weekly_periods: 周线数据周数（默认20周）
        monthly_periods: 月线数据月数（默认12月）

    Returns:
        包含基本信息、最新行情、日线/周线/月线 OHLCV 数据的字典
    """
    try:
        # 自动判断市场
        if stock_code.startswith("6"):
            market = "sh"
        elif stock_code.startswith(("0", "3")):
            market = "sz"
        else:
            market = "bj"

        # 日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")

        # 获取股票名称
        stock_name = _get_stock_name(stock_code)

        # 获取日线数据
        df_daily = _fetch_daily_data(stock_code, market, start_date, end_date, days)
        if df_daily is None or df_daily.empty:
            logger.error(f"无法获取 {stock_code} 日线数据")
            return None

        # 计算涨跌幅（如果没有）
        if "pct_change" not in df_daily.columns:
            df_daily["pct_change"] = df_daily["close"].pct_change() * 100

        # 获取周线数据（添加延迟避免API限制）
        import time
        weekly_ohlcv = []
        if include_weekly:
            time.sleep(0.5)  # 等待0.5秒
            df_weekly = _fetch_weekly_data(stock_code, market, weekly_periods)
            weekly_ohlcv = _df_to_ohlcv_list(df_weekly)

        # 获取月线数据
        monthly_ohlcv = []
        if include_monthly:
            time.sleep(0.5)  # 等待0.5秒
            df_monthly = _fetch_monthly_data(stock_code, market, monthly_periods)
            monthly_ohlcv = _df_to_ohlcv_list(df_monthly)

        # 构建返回数据
        latest_daily = df_daily.iloc[-1]

        # 获取最近 N 天的日线数据
        actual_recent_days = min(recent_days, len(df_daily))
        daily_ohlcv = _df_to_ohlcv_list(df_daily.tail(actual_recent_days))

        # 尝试获取实时行情（优先使用实时数据）
        realtime_quote = _fetch_realtime_quote(stock_code)

        # 如果有实时行情，使用实时数据；否则使用日线最后一条数据
        if realtime_quote and realtime_quote.get("price") is not None:
            latest_quote = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "open": round(float(realtime_quote["open"]), 2) if realtime_quote.get("open") else round(float(latest_daily["open"]), 2),
                "high": round(float(realtime_quote["high"]), 2) if realtime_quote.get("high") else round(float(latest_daily["high"]), 2),
                "low": round(float(realtime_quote["low"]), 2) if realtime_quote.get("low") else round(float(latest_daily["low"]), 2),
                "close": round(float(realtime_quote["price"]), 2),  # 实时价格
                "volume": int(realtime_quote.get("volume", 0)),
                "pct_change": round(float(realtime_quote.get("pct_change", 0)), 2),
                "is_realtime": True,  # 标记为实时数据
            }
            logger.info(f"[实时行情] {stock_code} 使用实时价格: {latest_quote['close']}")
        else:
            # 回退到日线数据
            latest_quote = {
                "date": latest_daily["date"].strftime("%Y-%m-%d") if hasattr(latest_daily["date"], "strftime") else str(latest_daily["date"]),
                "open": round(float(latest_daily["open"]), 2),
                "high": round(float(latest_daily["high"]), 2),
                "low": round(float(latest_daily["low"]), 2),
                "close": round(float(latest_daily["close"]), 2),
                "volume": int(latest_daily["volume"]),
                "pct_change": round(float(latest_daily["pct_change"]), 2) if pd.notna(latest_daily["pct_change"]) else 0,
                "is_realtime": False,  # 标记为历史数据
            }
            logger.warning(f"[日线数据] {stock_code} 无法获取实时行情，使用收盘价: {latest_quote['close']}")

        return {
            "basic_info": {
                "symbol": stock_code,
                "name": stock_name,
                "market": market,
            },
            "latest_quote": latest_quote,
            "daily_ohlcv": daily_ohlcv,
            "weekly_ohlcv": weekly_ohlcv,
            "monthly_ohlcv": monthly_ohlcv,
            # 保持向后兼容
            "recent_ohlcv": daily_ohlcv,
            "full_data": df_daily,
        }

    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _format_volume(volume: float) -> str:
    """格式化成交量为可读格式（M=百万, B=十亿）"""
    if volume >= 1000000000:  # 10亿 = 1B
        return f"{volume / 1000000000:.2f}B"
    elif volume >= 1000000:  # 100万 = 1M
        return f"{volume / 1000000:.1f}M"
    elif volume >= 1000:  # 1K
        return f"{volume / 1000:.1f}K"
    else:
        return f"{volume:.0f}"


def format_ohlcv_for_llm(recent_ohlcv: List[Dict[str, Any]]) -> str:
    """将 OHLCV 数据格式化为 LLM 可读的表格

    Args:
        recent_ohlcv: 近期 OHLCV 数据列表

    Returns:
        格式化的表格字符串
    """
    table = "Date       | Open   | High   | Low    | Close  | Volume      | Change%\n"
    table += "-" * 78 + "\n"
    for row in recent_ohlcv:
        vol_str = _format_volume(row['volume'])
        table += (
            f"{row['date']} | "
            f"{row['open']:6.2f} | "
            f"{row['high']:6.2f} | "
            f"{row['low']:6.2f} | "
            f"{row['close']:6.2f} | "
            f"{vol_str:>11} | "
            f"{row['pct_change']:+6.2f}%\n"
        )
    return table


def build_finance_llm_prompt(
    stock_name: str,
    stock_code: str,
    latest_quote: Dict[str, Any],
    recent_ohlcv: List[Dict[str, Any]],
    analysis_type: str = "full",
    weekly_ohlcv: Optional[List[Dict[str, Any]]] = None,
    monthly_ohlcv: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """构建 Finance LLM 分析提示词

    Args:
        stock_name: 股票名称
        stock_code: 股票代码
        latest_quote: 最新行情
        recent_ohlcv: 近期日线 OHLCV 数据
        analysis_type: 分析类型 - "full" 完整报告, "brief" 简要分析
        weekly_ohlcv: 周线 OHLCV 数据（可选）
        monthly_ohlcv: 月线 OHLCV 数据（可选）

    Returns:
        提示词字符串
    """
    daily_table = format_ohlcv_for_llm(recent_ohlcv)

    # 构建周线和月线数据部分
    weekly_section = ""
    if weekly_ohlcv:
        weekly_table = format_ohlcv_for_llm(weekly_ohlcv)
        weekly_section = f"""
## Weekly Price History (Last {len(weekly_ohlcv)} Weeks)
{weekly_table}
"""

    monthly_section = ""
    if monthly_ohlcv:
        monthly_table = format_ohlcv_for_llm(monthly_ohlcv)
        monthly_section = f"""
## Monthly Price History (Last {len(monthly_ohlcv)} Months)
{monthly_table}
"""

    # 从 md 文件加载 prompt 模板
    if analysis_type == "brief":
        template = _load_prompt_template("finance_llm_brief.md")
    else:
        template = _load_prompt_template("finance_llm_full.md")

    if not template:
        logger.error(f"无法加载 {analysis_type} prompt 模板")
        return ""

    # 替换模板变量
    prompt = template.format(
        stock_name=stock_name,
        stock_code=stock_code,
        latest_price=latest_quote.get("close", "N/A"),
        pct_change=latest_quote.get("pct_change", "N/A"),
        volume=_format_volume(latest_quote.get("volume", 0)),
        daily_count=len(recent_ohlcv),
        daily_table=daily_table,
        weekly_section=weekly_section,
        monthly_section=monthly_section,
    )

    return prompt
