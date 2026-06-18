"""股票数据获取 Skill - 获取原始数据并由 Finance LLM 分析技术指标"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama

from .base import BaseSkill, SkillOutput
from .stock_data_fetcher import (
    fetch_stock_raw_data,
    build_finance_llm_prompt,
)
from ..config import ollama_config

logger = logging.getLogger(__name__)


class StockRetrieverSkill(BaseSkill):
    """股票数据获取 Skill

    获取原始 OHLCV 数据，由 Finance LLM 分析计算技术指标。
    不再通过 Python 代码预计算指标，而是让专业金融模型自行分析。
    """

    name = "stock_retriever"
    description = "获取股票历史数据，由 Finance LLM 分析技术指标(MA/MACD/KDJ/RSI/布林带/K线形态等)，返回结构化分析结果"

    # Finance LLM 模型配置
    FINANCE_MODEL = "martain7r/finance-llama-8b:q4_k_m"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如 '000001' (平安银行), '600519' (贵州茅台)"
                },
                "days": {
                    "type": "integer",
                    "description": "获取过去多少天的数据 (默认60天)"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["full", "brief"],
                    "description": "分析类型: full=完整报告, brief=简要分析 (默认full)"
                }
            },
            "required": ["symbol"]
        }

    async def execute(
        self,
        symbol: str,
        days: int = 60,
        analysis_type: str = "full",
        **kwargs
    ) -> SkillOutput:
        """获取股票数据并生成 Finance LLM 分析报告

        Args:
            symbol: 股票代码
            days: 历史数据天数
            analysis_type: 分析类型 (full/brief)

        Returns:
            SkillOutput 包含原始数据和 LLM 分析结果
        """
        try:
            # 1. 获取原始行情数据
            logger.info(f"📊 [{symbol}] 获取原始行情数据...")
            stock_data = await fetch_stock_raw_data(
                stock_code=symbol,
                days=days,
                recent_days=60,  # 日线数据60天
                weekly_periods=20,  # 周线数据20周
                monthly_periods=12,  # 月线数据12月
            )

            if not stock_data:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"无法获取股票 {symbol} 的数据"
                )

            basic_info = stock_data["basic_info"]
            latest_quote = stock_data["latest_quote"]
            recent_ohlcv = stock_data["recent_ohlcv"]
            weekly_ohlcv = stock_data.get("weekly_ohlcv", [])
            monthly_ohlcv = stock_data.get("monthly_ohlcv", [])

            # 2. 调用 Finance LLM 分析
            logger.info(f"🤖 [{symbol}] 调用 Finance LLM 分析技术指标...")
            prompt = build_finance_llm_prompt(
                stock_name=basic_info["name"],
                stock_code=basic_info["symbol"],
                latest_quote=latest_quote,
                recent_ohlcv=recent_ohlcv,
                analysis_type=analysis_type,
                weekly_ohlcv=weekly_ohlcv,
                monthly_ohlcv=monthly_ohlcv,
            )

            llm = ChatOllama(
                model=self.FINANCE_MODEL,
                base_url=ollama_config.base_url,
                temperature=0.3,
            )
            response = await llm.ainvoke(prompt)
            analysis_report = response.content

            logger.info(f"✅ [{symbol}] 分析完成")

            # 3. 构建返回结果
            result = {
                "basic_info": {
                    "symbol": basic_info["symbol"],
                    "name": basic_info["name"],
                    "market": basic_info["market"],
                    "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_days": len(recent_ohlcv),
                },
                "latest_quote": latest_quote,
                "recent_ohlcv": recent_ohlcv,
                "analysis_report": analysis_report,
            }

            return SkillOutput(
                success=True,
                result=json.dumps(result, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            logger.error(f"❌ [{symbol}] 数据获取失败: {e}")
            import traceback
            traceback.print_exc()
            return SkillOutput(
                success=False,
                result=None,
                error=f"股票数据获取失败: {str(e)}"
            )

    async def get_raw_data(self, symbol: str, days: int = 60) -> Optional[Dict[str, Any]]:
        """仅获取原始数据（不调用 LLM）

        供其他模块直接使用原始数据。

        Args:
            symbol: 股票代码
            days: 历史数据天数

        Returns:
            原始行情数据字典，或 None
        """
        return await fetch_stock_raw_data(stock_code=symbol, days=days)

    async def get_index_data(self) -> Dict[str, Any]:
        """获取主要指数实时数据（上证、深证、创业板、恒指）"""
        import akshare as ak

        indices = {}

        # A股指数代码映射 (新浪格式)
        a_share_indices = {
            "上证指数": "sh000001",
            "深证成指": "sz399001",
            "创业板指": "sz399006",
        }

        try:
            # 使用新浪实时行情API获取A股指数
            df = ak.stock_zh_index_spot_sina()
            if df is not None and not df.empty:
                for name, code in a_share_indices.items():
                    try:
                        row = df[df["代码"] == code]
                        if not row.empty:
                            row = row.iloc[0]
                            indices[name] = {
                                "code": code,
                                "close": round(float(row["最新价"]), 2),
                                "change": round(float(row["涨跌额"]), 2),
                                "pct_change": round(float(row["涨跌幅"]), 2),
                                "high": round(float(row["最高"]), 2),
                                "low": round(float(row["最低"]), 2),
                                "open": round(float(row["今开"]), 2),
                                "prev_close": round(float(row["昨收"]), 2),
                                "volume": round(float(row["成交量"]) / 100000000, 2),
                                "amount": round(float(row["成交额"]) / 100000000, 2),
                            }
                    except Exception as e:
                        logger.warning(f"解析 {name} 实时数据失败: {e}")
                        indices[name] = {"error": str(e)}
        except Exception as e:
            logger.warning(f"获取A股指数实时行情失败: {e}")
            # 回退到历史数据
            for name, code in a_share_indices.items():
                try:
                    df = ak.stock_zh_index_daily(symbol=code)
                    if df is not None and not df.empty:
                        latest = df.iloc[-1]
                        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                        change = latest["close"] - prev["close"]
                        pct_change = (change / prev["close"]) * 100 if prev["close"] > 0 else 0
                        indices[name] = {
                            "code": code,
                            "close": round(float(latest["close"]), 2),
                            "change": round(float(change), 2),
                            "pct_change": round(float(pct_change), 2),
                            "high": round(float(latest["high"]), 2),
                            "low": round(float(latest["low"]), 2),
                            "open": round(float(latest["open"]), 2),
                            "note": "历史数据(实时接口不可用)"
                        }
                except Exception as ex:
                    indices[name] = {"error": str(ex)}

        # 恒生指数
        try:
            df_hk = ak.stock_hk_index_spot_sina()
            if df_hk is not None and not df_hk.empty:
                hsi_row = df_hk[df_hk["代码"] == "HSI"]
                if not hsi_row.empty:
                    row = hsi_row.iloc[0]
                    indices["恒生指数"] = {
                        "code": "HSI",
                        "close": round(float(row["最新价"]), 2),
                        "change": round(float(row["涨跌额"]), 2),
                        "pct_change": round(float(row["涨跌幅"]), 2),
                        "high": round(float(row["最高"]), 2),
                        "low": round(float(row["最低"]), 2),
                        "open": round(float(row["今开"]), 2),
                        "prev_close": round(float(row["昨收"]), 2),
                    }
        except Exception as e:
            logger.warning(f"获取恒生指数实时行情失败: {e}")
            try:
                df = ak.stock_hk_index_daily_sina(symbol="HSI")
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                    change = float(latest["close"]) - float(prev["close"])
                    pct_change = (change / float(prev["close"])) * 100 if float(prev["close"]) > 0 else 0
                    indices["恒生指数"] = {
                        "code": "HSI",
                        "close": round(float(latest["close"]), 2),
                        "change": round(float(change), 2),
                        "pct_change": round(float(pct_change), 2),
                        "high": round(float(latest["high"]), 2),
                        "low": round(float(latest["low"]), 2),
                        "open": round(float(latest["open"]), 2),
                        "note": "历史数据(实时接口不可用)"
                    }
            except Exception as ex:
                indices["恒生指数"] = {"error": str(ex)}

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "indices": indices
        }
