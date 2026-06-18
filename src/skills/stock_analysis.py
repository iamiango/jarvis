"""股票分析 Skill - 使用 AKShare 获取数据并进行 K 线技术分析"""
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base import BaseSkill, SkillOutput


class StockAnalysisSkill(BaseSkill):
    """股票分析 Skill - 获取股票数据并进行技术分析"""

    name = "stock_analysis"
    description = "获取股票历史数据并进行K线技术分析，包括MACD、KDJ、均线、经典K线形态识别等，生成完整分析报告"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码，如 '000001' (平安银行), '600519' (贵州茅台)"
                },
                "market": {
                    "type": "string",
                    "enum": ["sh", "sz", "bj"],
                    "description": "市场类型: sh=上海, sz=深圳, bj=北京 (默认根据代码自动判断)"
                },
                "days": {
                    "type": "integer",
                    "description": "获取过去多少天的数据 (默认60天)"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["full", "macd", "kdj", "ma", "pattern", "trend"],
                    "description": "分析类型: full=完整分析, macd=MACD分析, kdj=KDJ分析, ma=均线分析, pattern=K线形态, trend=趋势分析"
                }
            },
            "required": ["symbol"]
        }

    async def execute(
        self,
        symbol: str,
        market: Optional[str] = None,
        days: int = 60,
        analysis_type: str = "full",
        **kwargs
    ) -> SkillOutput:
        try:
            # 1. 获取股票数据
            df, stock_name = await self._fetch_stock_data(symbol, market, days)
            if df is None or df.empty:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"无法获取股票 {symbol} 的数据"
                )

            # 2. 计算技术指标
            df = self._calculate_indicators(df)

            # 3. 生成分析报告
            report = self._generate_professional_report(df, symbol, stock_name, analysis_type)

            return SkillOutput(success=True, result=report)

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"股票分析失败: {str(e)}"
            )

    async def _fetch_stock_data(
        self, symbol: str, market: Optional[str], days: int
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """使用 AKShare 获取股票历史数据（优先使用新浪数据源）"""
        import akshare as ak
        import time

        # 自动判断市场
        if market is None:
            if symbol.startswith("6"):
                market = "sh"
            elif symbol.startswith(("0", "3")):
                market = "sz"
            else:
                market = "bj"

        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days + 60)).strftime("%Y%m%d")

        # 获取股票名称
        stock_name = symbol
        try:
            stock_info = ak.stock_individual_info_em(symbol=symbol)
            if stock_info is not None and not stock_info.empty:
                name_row = stock_info[stock_info['item'] == '股票简称']
                if not name_row.empty:
                    stock_name = name_row['value'].values[0]
        except:
            pass

        # 方法1: 使用新浪数据源（更稳定）
        try:
            sina_symbol = f"{market}{symbol}"
            df = ak.stock_zh_a_daily(
                symbol=sina_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            if df is not None and not df.empty:
                # 新浪数据列名标准化
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "close": "close",
                    "high": "high",
                    "low": "low",
                    "volume": "volume",
                    "amount": "amount",
                    "outstanding_share": "outstanding_share",
                    "turnover": "turnover"
                })
                df["date"] = pd.to_datetime(df["date"])
                # 计算涨跌幅
                df["pct_change"] = df["close"].pct_change() * 100
                df["change"] = df["close"].diff()
                df = df.sort_values("date").tail(days).reset_index(drop=True)
                print(f"[新浪数据源] 成功获取 {stock_name}({symbol}) {len(df)} 条数据")
                return df, stock_name
        except Exception as e:
            print(f"[新浪数据源] 获取失败: {e}")

        # 方法2: 使用东方财富数据源（备选）
        max_retries = 2
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )

                if df is None or df.empty:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    break

                # 东方财富数据列名标准化
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "换手率": "turnover"
                })

                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").tail(days).reset_index(drop=True)
                print(f"[东方财富数据源] 成功获取 {stock_name}({symbol}) {len(df)} 条数据")
                return df, stock_name

            except Exception as e:
                print(f"[东方财富数据源] 获取失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return None, ""

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()

        # 均线系统
        for period in [5, 10, 20, 30, 60]:
            df[f"ma{period}"] = df["close"].rolling(window=period).mean()

        # EMA 均线
        for period in [12, 26]:
            df[f"ema{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD
        df = self._calculate_macd(df)

        # KDJ
        df = self._calculate_kdj(df)

        # RSI
        df = self._calculate_rsi(df)

        # 布林带
        df = self._calculate_bollinger(df)

        # 成交量均线
        df["vol_ma5"] = df["volume"].rolling(window=5).mean()
        df["vol_ma10"] = df["volume"].rolling(window=10).mean()

        # 量比
        df["volume_ratio"] = df["volume"] / df["vol_ma5"]

        # ATR (真实波幅)
        df = self._calculate_atr(df)

        # K线形态
        df = self._identify_patterns(df)

        # 支撑位和压力位
        df = self._calculate_support_resistance(df)

        return df

    def _calculate_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """计算 MACD 指标"""
        df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
        df["dif"] = df["ema_fast"] - df["ema_slow"]
        df["dea"] = df["dif"].ewm(span=signal, adjust=False).mean()
        df["macd"] = (df["dif"] - df["dea"]) * 2

        # MACD 柱状图变化
        df["macd_growing"] = df["macd"] > df["macd"].shift(1)

        return df

    def _calculate_kdj(self, df: pd.DataFrame, n=9, m1=3, m2=3) -> pd.DataFrame:
        """计算 KDJ 指标"""
        low_min = df["low"].rolling(window=n).min()
        high_max = df["high"].rolling(window=n).max()

        rsv = (df["close"] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)

        df["k"] = rsv.ewm(alpha=1/m1, adjust=False).mean()
        df["d"] = df["k"].ewm(alpha=1/m2, adjust=False).mean()
        df["j"] = 3 * df["k"] - 2 * df["d"]

        return df

    def _calculate_rsi(self, df: pd.DataFrame, periods=[6, 12, 24]) -> pd.DataFrame:
        """计算 RSI 指标"""
        for period in periods:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f"rsi{period}"] = 100 - (100 / (1 + rs))
        return df

    def _calculate_bollinger(self, df: pd.DataFrame, window=20, num_std=2) -> pd.DataFrame:
        """计算布林带"""
        df["boll_mid"] = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        df["boll_upper"] = df["boll_mid"] + num_std * std
        df["boll_lower"] = df["boll_mid"] - num_std * std
        df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"] * 100
        return df

    def _calculate_atr(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """计算 ATR"""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period).mean()
        df["atr_percent"] = df["atr"] / df["close"] * 100
        return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算支撑位和压力位"""
        # 使用近期高低点
        df["resistance1"] = df["high"].rolling(window=20).max()
        df["support1"] = df["low"].rolling(window=20).min()

        # 枢轴点
        df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
        df["r1"] = 2 * df["pivot"] - df["low"]
        df["s1"] = 2 * df["pivot"] - df["high"]

        return df

    def _identify_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别 K 线形态"""
        df["pattern"] = ""
        df["pattern_type"] = ""  # bullish, bearish, neutral

        df["body"] = df["close"] - df["open"]
        df["body_abs"] = abs(df["body"])
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["range"] = df["high"] - df["low"]

        patterns = []
        pattern_types = []

        for i in range(len(df)):
            row = df.iloc[i]
            pattern_list = []
            p_type = "neutral"

            body = row["body"]
            body_abs = row["body_abs"]
            upper = row["upper_shadow"]
            lower = row["lower_shadow"]
            total_range = row["range"]

            if total_range == 0:
                patterns.append("")
                pattern_types.append("neutral")
                continue

            body_ratio = body_abs / total_range

            # === 单日形态 ===

            # 十字星系列
            if body_ratio < 0.1:
                if upper > body_abs * 2 and lower > body_abs * 2:
                    pattern_list.append("十字星")
                    p_type = "neutral"
                elif upper > body_abs * 3 and lower < body_abs:
                    pattern_list.append("墓碑十字(看跌)")
                    p_type = "bearish"
                elif lower > body_abs * 3 and upper < body_abs:
                    pattern_list.append("蜻蜓十字(看涨)")
                    p_type = "bullish"
                elif upper < body_abs and lower < body_abs:
                    pattern_list.append("一字线")
                    p_type = "neutral"

            # 锤子线/上吊线
            elif body_ratio < 0.35 and lower > body_abs * 2 and upper < body_abs * 0.5:
                if i >= 5:
                    recent_trend = df.iloc[i-5:i]["close"].mean()
                    if row["close"] < recent_trend:
                        pattern_list.append("锤子线(底部看涨)")
                        p_type = "bullish"
                    else:
                        pattern_list.append("上吊线(顶部看跌)")
                        p_type = "bearish"

            # 倒锤子/射击之星
            elif body_ratio < 0.35 and upper > body_abs * 2 and lower < body_abs * 0.5:
                if i >= 5:
                    recent_trend = df.iloc[i-5:i]["close"].mean()
                    if row["close"] < recent_trend:
                        pattern_list.append("倒锤子(底部看涨)")
                        p_type = "bullish"
                    else:
                        pattern_list.append("射击之星(顶部看跌)")
                        p_type = "bearish"

            # 大阳线/大阴线
            elif body_ratio > 0.65:
                if body > 0:
                    if body_ratio > 0.85:
                        pattern_list.append("光头光脚大阳线(强势)")
                    else:
                        pattern_list.append("大阳线(看涨)")
                    p_type = "bullish"
                else:
                    if body_ratio > 0.85:
                        pattern_list.append("光头光脚大阴线(弱势)")
                    else:
                        pattern_list.append("大阴线(看跌)")
                    p_type = "bearish"

            # 纺锤线
            elif 0.2 < body_ratio < 0.4 and upper > body_abs * 0.5 and lower > body_abs * 0.5:
                pattern_list.append("纺锤线(犹豫)")
                p_type = "neutral"

            # === 多日形态 ===
            if i >= 2:
                prev1 = df.iloc[i-1]
                prev2 = df.iloc[i-2]

                # 早晨之星
                if (prev2["body"] < 0 and abs(prev2["body"]) > prev2["range"] * 0.5 and
                    abs(prev1["body"]) < prev1["range"] * 0.3 and
                    body > 0 and body > total_range * 0.5 and
                    row["close"] > prev2["open"]):
                    pattern_list.append("★早晨之星(强烈看涨)")
                    p_type = "bullish"

                # 黄昏之星
                if (prev2["body"] > 0 and prev2["body"] > prev2["range"] * 0.5 and
                    abs(prev1["body"]) < prev1["range"] * 0.3 and
                    body < 0 and abs(body) > total_range * 0.5 and
                    row["close"] < prev2["open"]):
                    pattern_list.append("★黄昏之星(强烈看跌)")
                    p_type = "bearish"

                # 红三兵
                if (prev2["body"] > 0 and prev1["body"] > 0 and body > 0 and
                    prev1["close"] > prev2["close"] and row["close"] > prev1["close"] and
                    prev1["open"] > prev2["open"] and row["open"] > prev1["open"]):
                    pattern_list.append("★红三兵(强势上涨)")
                    p_type = "bullish"

                # 三只乌鸦
                if (prev2["body"] < 0 and prev1["body"] < 0 and body < 0 and
                    prev1["close"] < prev2["close"] and row["close"] < prev1["close"]):
                    pattern_list.append("★三只乌鸦(强势下跌)")
                    p_type = "bearish"

                # 吞没形态
                if (prev1["body"] < 0 and body > 0 and
                    row["open"] < prev1["close"] and row["close"] > prev1["open"]):
                    pattern_list.append("看涨吞没")
                    p_type = "bullish"

                if (prev1["body"] > 0 and body < 0 and
                    row["open"] > prev1["close"] and row["close"] < prev1["open"]):
                    pattern_list.append("看跌吞没")
                    p_type = "bearish"

                # 乌云盖顶
                if (prev1["body"] > 0 and body < 0 and
                    row["open"] > prev1["high"] and
                    row["close"] < (prev1["open"] + prev1["close"]) / 2):
                    pattern_list.append("乌云盖顶(看跌)")
                    p_type = "bearish"

                # 刺透形态
                if (prev1["body"] < 0 and body > 0 and
                    row["open"] < prev1["low"] and
                    row["close"] > (prev1["open"] + prev1["close"]) / 2):
                    pattern_list.append("刺透形态(看涨)")
                    p_type = "bullish"

            patterns.append(", ".join(pattern_list) if pattern_list else "")
            pattern_types.append(p_type)

        df["pattern"] = patterns
        df["pattern_type"] = pattern_types
        return df

    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """深度趋势分析"""
        latest = df.iloc[-1]
        prev_5 = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
        prev_20 = df.iloc[-20] if len(df) >= 20 else df.iloc[0]

        # 均线排列
        ma_values = {
            "ma5": latest.get("ma5", 0),
            "ma10": latest.get("ma10", 0),
            "ma20": latest.get("ma20", 0),
            "ma30": latest.get("ma30", 0),
            "ma60": latest.get("ma60", 0),
        }

        # 判断均线多空排列
        valid_mas = [v for v in ma_values.values() if pd.notna(v) and v > 0]
        if len(valid_mas) >= 3:
            if valid_mas == sorted(valid_mas, reverse=True):
                ma_arrangement = "多头排列 ↑"
                ma_score = 2
            elif valid_mas == sorted(valid_mas):
                ma_arrangement = "空头排列 ↓"
                ma_score = -2
            else:
                ma_arrangement = "交叉缠绕 ≈"
                ma_score = 0
        else:
            ma_arrangement = "数据不足"
            ma_score = 0

        # 价格与均线关系
        price = latest["close"]
        above_ma = sum(1 for k, v in ma_values.items() if pd.notna(v) and price > v)
        below_ma = sum(1 for k, v in ma_values.items() if pd.notna(v) and price < v)

        # 计算涨跌幅
        pct_1d = latest.get("pct_change", 0)
        pct_5d = ((latest["close"] / prev_5["close"]) - 1) * 100 if prev_5["close"] > 0 else 0
        pct_20d = ((latest["close"] / prev_20["close"]) - 1) * 100 if prev_20["close"] > 0 else 0

        # 波动率
        volatility_20d = df["pct_change"].tail(20).std() if "pct_change" in df.columns else 0

        # 趋势强度 (ADX 简化版)
        recent_highs = df["high"].tail(14).max()
        recent_lows = df["low"].tail(14).min()
        trend_range = (recent_highs - recent_lows) / latest["close"] * 100

        return {
            "ma_arrangement": ma_arrangement,
            "ma_score": ma_score,
            "ma_values": ma_values,
            "above_ma_count": above_ma,
            "below_ma_count": below_ma,
            "pct_1d": round(pct_1d, 2),
            "pct_5d": round(pct_5d, 2),
            "pct_20d": round(pct_20d, 2),
            "volatility_20d": round(volatility_20d, 2),
            "trend_range": round(trend_range, 2),
            "price": round(price, 2),
        }

    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """深度 MACD 分析"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        dif = latest["dif"]
        dea = latest["dea"]
        macd = latest["macd"]

        # 金叉死叉判断
        if latest["dif"] > latest["dea"] and prev["dif"] <= prev["dea"]:
            cross_signal = "金叉 ↑"
            cross_type = "bullish"
        elif latest["dif"] < latest["dea"] and prev["dif"] >= prev["dea"]:
            cross_signal = "死叉 ↓"
            cross_type = "bearish"
        elif latest["dif"] > latest["dea"]:
            cross_signal = "多头运行"
            cross_type = "bullish"
        else:
            cross_signal = "空头运行"
            cross_type = "bearish"

        # 零轴位置
        if dif > 0 and dea > 0:
            zero_axis = "零轴之上(强势区)"
        elif dif < 0 and dea < 0:
            zero_axis = "零轴之下(弱势区)"
        else:
            zero_axis = "零轴附近(转换区)"

        # MACD 柱状图趋势
        macd_trend = "放量" if latest["macd_growing"] else "缩量"
        if macd > 0:
            macd_color = "红柱" + macd_trend
        else:
            macd_color = "绿柱" + macd_trend

        # 背离检测
        divergence = self._detect_divergence(df)

        # MACD 历史数据
        macd_history = df[["date", "dif", "dea", "macd"]].tail(5)

        return {
            "dif": round(dif, 4),
            "dea": round(dea, 4),
            "macd": round(macd, 4),
            "cross_signal": cross_signal,
            "cross_type": cross_type,
            "zero_axis": zero_axis,
            "macd_color": macd_color,
            "divergence": divergence,
            "history": macd_history,
        }

    def _analyze_kdj(self, df: pd.DataFrame) -> Dict[str, Any]:
        """深度 KDJ 分析"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        k, d, j = latest["k"], latest["d"], latest["j"]

        # 金叉死叉
        if k > d and prev["k"] <= prev["d"]:
            cross_signal = "金叉 ↑"
            cross_type = "bullish"
        elif k < d and prev["k"] >= prev["d"]:
            cross_signal = "死叉 ↓"
            cross_type = "bearish"
        elif k > d:
            cross_signal = "多头排列"
            cross_type = "bullish"
        else:
            cross_signal = "空头排列"
            cross_type = "bearish"

        # 超买超卖区域
        if j > 100:
            zone = "极度超买区 ⚠️"
            zone_advice = "警惕回调风险"
        elif k > 80:
            zone = "超买区"
            zone_advice = "注意高位风险"
        elif j < 0:
            zone = "极度超卖区 💡"
            zone_advice = "关注反弹机会"
        elif k < 20:
            zone = "超卖区"
            zone_advice = "可能超跌反弹"
        elif 40 <= k <= 60:
            zone = "中性区"
            zone_advice = "观望为主"
        else:
            zone = "正常区"
            zone_advice = "趋势跟随"

        # J 值钝化
        j_blunt = ""
        if j > 100:
            j_high_count = (df["j"].tail(5) > 100).sum()
            if j_high_count >= 3:
                j_blunt = "J值高位钝化(强势特征)"
        elif j < 0:
            j_low_count = (df["j"].tail(5) < 0).sum()
            if j_low_count >= 3:
                j_blunt = "J值低位钝化(弱势特征)"

        return {
            "k": round(k, 2),
            "d": round(d, 2),
            "j": round(j, 2),
            "cross_signal": cross_signal,
            "cross_type": cross_type,
            "zone": zone,
            "zone_advice": zone_advice,
            "j_blunt": j_blunt,
        }

    def _analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """RSI 分析"""
        latest = df.iloc[-1]

        rsi6 = latest.get("rsi6", 50)
        rsi12 = latest.get("rsi12", 50)
        rsi24 = latest.get("rsi24", 50)

        # RSI 区域判断
        if rsi6 > 80:
            zone = "超买区"
            signal = "卖出信号"
        elif rsi6 < 20:
            zone = "超卖区"
            signal = "买入信号"
        elif rsi6 > 50:
            zone = "强势区"
            signal = "持股待涨"
        else:
            zone = "弱势区"
            signal = "观望或减仓"

        # RSI 背离
        rsi_trend = "上升" if rsi6 > rsi12 > rsi24 else "下降" if rsi6 < rsi12 < rsi24 else "震荡"

        return {
            "rsi6": round(rsi6, 2),
            "rsi12": round(rsi12, 2),
            "rsi24": round(rsi24, 2),
            "zone": zone,
            "signal": signal,
            "trend": rsi_trend,
        }

    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """成交量分析"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        volume = latest["volume"]
        vol_ma5 = latest.get("vol_ma5", volume)
        vol_ma10 = latest.get("vol_ma10", volume)
        amount = latest.get("amount", 0)
        turnover = latest.get("turnover", 0)

        # 量比
        volume_ratio = volume / vol_ma5 if vol_ma5 > 0 else 1

        # 量能判断
        if volume_ratio > 2:
            vol_status = "放量(量比{:.2f})".format(volume_ratio)
            vol_signal = "资金活跃"
        elif volume_ratio > 1.5:
            vol_status = "温和放量(量比{:.2f})".format(volume_ratio)
            vol_signal = "关注度提升"
        elif volume_ratio < 0.5:
            vol_status = "缩量(量比{:.2f})".format(volume_ratio)
            vol_signal = "交投清淡"
        else:
            vol_status = "量能平稳(量比{:.2f})".format(volume_ratio)
            vol_signal = "正常交易"

        # 量价配合
        price_up = latest["close"] > prev["close"]
        vol_up = volume > prev["volume"]

        if price_up and vol_up:
            vol_price = "量价齐升(健康上涨)"
        elif price_up and not vol_up:
            vol_price = "价涨量缩(上涨乏力)"
        elif not price_up and vol_up:
            vol_price = "价跌量增(恐慌抛售)"
        else:
            vol_price = "价跌量缩(下跌趋缓)"

        return {
            "volume": int(volume),
            "vol_ma5": int(vol_ma5) if pd.notna(vol_ma5) else 0,
            "vol_ma10": int(vol_ma10) if pd.notna(vol_ma10) else 0,
            "amount": round(amount / 100000000, 2) if amount > 0 else 0,  # 亿元
            "turnover": round(turnover, 2),
            "volume_ratio": round(volume_ratio, 2),
            "vol_status": vol_status,
            "vol_signal": vol_signal,
            "vol_price": vol_price,
        }

    def _analyze_bollinger(self, df: pd.DataFrame) -> Dict[str, Any]:
        """布林带分析"""
        latest = df.iloc[-1]

        price = latest["close"]
        upper = latest.get("boll_upper", price)
        mid = latest.get("boll_mid", price)
        lower = latest.get("boll_lower", price)
        width = latest.get("boll_width", 0)

        # 计算位置百分比 (0-100, 0=下轨, 100=上轨)
        if upper != lower:
            position_pct = (price - lower) / (upper - lower) * 100
        else:
            position_pct = 50
        position_pct = max(0, min(100, position_pct))

        # 价格位置
        if price > upper:
            position = "突破上轨(超买)"
            signal = "注意回调风险"
        elif price > mid:
            pct = (price - mid) / (upper - mid) * 100 if upper != mid else 50
            position = f"中轨上方({pct:.0f}%位置)"
            signal = "偏强运行"
        elif price > lower:
            pct = (price - lower) / (mid - lower) * 100 if mid != lower else 50
            position = f"中轨下方({pct:.0f}%位置)"
            signal = "偏弱运行"
        else:
            position = "跌破下轨(超卖)"
            signal = "关注反弹机会"

        # 带宽判断
        if width > 20:
            width_status = "带宽扩张(波动加大)"
        elif width < 10:
            width_status = "带宽收窄(变盘在即)"
        else:
            width_status = "带宽正常"

        return {
            "upper": round(upper, 2),
            "mid": round(mid, 2),
            "lower": round(lower, 2),
            "width": round(width, 2),
            "position": position,
            "position_pct": position_pct,
            "signal": signal,
            "width_status": width_status,
        }

    def _detect_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检测背离"""
        if len(df) < 30:
            return {"type": "数据不足", "description": ""}

        recent = df.tail(30)

        # 找出价格高点和低点
        price_highs_idx = []
        price_lows_idx = []
        macd_at_highs = []
        macd_at_lows = []

        for i in range(2, len(recent) - 2):
            # 局部高点
            if (recent.iloc[i]["high"] > recent.iloc[i-1]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i-2]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i+1]["high"] and
                recent.iloc[i]["high"] > recent.iloc[i+2]["high"]):
                price_highs_idx.append(i)
                macd_at_highs.append(recent.iloc[i]["dif"])

            # 局部低点
            if (recent.iloc[i]["low"] < recent.iloc[i-1]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i-2]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i+1]["low"] and
                recent.iloc[i]["low"] < recent.iloc[i+2]["low"]):
                price_lows_idx.append(i)
                macd_at_lows.append(recent.iloc[i]["dif"])

        # 顶背离检测
        if len(price_highs_idx) >= 2:
            if (recent.iloc[price_highs_idx[-1]]["high"] > recent.iloc[price_highs_idx[-2]]["high"] and
                macd_at_highs[-1] < macd_at_highs[-2]):
                return {
                    "type": "顶背离 ⚠️",
                    "description": "价格创新高但MACD未创新高，警惕高位回调"
                }

        # 底背离检测
        if len(price_lows_idx) >= 2:
            if (recent.iloc[price_lows_idx[-1]]["low"] < recent.iloc[price_lows_idx[-2]]["low"] and
                macd_at_lows[-1] > macd_at_lows[-2]):
                return {
                    "type": "底背离 💡",
                    "description": "价格创新低但MACD未创新低，关注反弹机会"
                }

        return {"type": "无明显背离", "description": ""}

    def _calculate_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算综合评分"""
        latest = df.iloc[-1]
        scores = {}
        total_score = 0

        # 1. 趋势得分 (30分)
        trend = self._analyze_trend(df)
        trend_score = 15  # 基础分
        trend_score += trend["ma_score"] * 5
        if trend["pct_5d"] > 5:
            trend_score += 5
        elif trend["pct_5d"] < -5:
            trend_score -= 5
        trend_score = max(0, min(30, trend_score))
        scores["趋势"] = trend_score
        total_score += trend_score

        # 2. MACD 得分 (20分)
        macd = self._analyze_macd(df)
        macd_score = 10
        if macd["cross_type"] == "bullish":
            macd_score += 5
        else:
            macd_score -= 5
        if "金叉" in macd["cross_signal"]:
            macd_score += 5
        elif "死叉" in macd["cross_signal"]:
            macd_score -= 5
        macd_score = max(0, min(20, macd_score))
        scores["MACD"] = macd_score
        total_score += macd_score

        # 3. KDJ 得分 (15分)
        kdj = self._analyze_kdj(df)
        kdj_score = 7
        if kdj["cross_type"] == "bullish":
            kdj_score += 4
        else:
            kdj_score -= 4
        if "超卖" in kdj["zone"]:
            kdj_score += 4
        elif "超买" in kdj["zone"]:
            kdj_score -= 4
        kdj_score = max(0, min(15, kdj_score))
        scores["KDJ"] = kdj_score
        total_score += kdj_score

        # 4. 量能得分 (15分)
        volume = self._analyze_volume(df)
        vol_score = 7
        if "量价齐升" in volume["vol_price"]:
            vol_score += 8
        elif "价涨量缩" in volume["vol_price"]:
            vol_score += 3
        elif "价跌量增" in volume["vol_price"]:
            vol_score -= 5
        vol_score = max(0, min(15, vol_score))
        scores["量能"] = vol_score
        total_score += vol_score

        # 5. 形态得分 (20分)
        pattern_score = 10
        recent_patterns = df["pattern_type"].tail(3).tolist()
        bullish_count = recent_patterns.count("bullish")
        bearish_count = recent_patterns.count("bearish")
        pattern_score += (bullish_count - bearish_count) * 3
        pattern_score = max(0, min(20, pattern_score))
        scores["形态"] = pattern_score
        total_score += pattern_score

        # 评级
        if total_score >= 80:
            rating = "强烈看多 ⭐⭐⭐⭐⭐"
        elif total_score >= 65:
            rating = "看多 ⭐⭐⭐⭐"
        elif total_score >= 50:
            rating = "中性偏多 ⭐⭐⭐"
        elif total_score >= 35:
            rating = "中性偏空 ⭐⭐"
        elif total_score >= 20:
            rating = "看空 ⭐"
        else:
            rating = "强烈看空 ⚠️"

        return {
            "total": total_score,
            "scores": scores,
            "rating": rating,
        }

    def _generate_suggestions(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """生成投资建议"""
        suggestions = []

        trend = self._analyze_trend(df)
        macd = self._analyze_macd(df)
        kdj = self._analyze_kdj(df)
        volume = self._analyze_volume(df)
        boll = self._analyze_bollinger(df)
        score = self._calculate_score(df)

        # 趋势建议
        if "多头" in trend["ma_arrangement"]:
            suggestions.append({
                "type": "趋势",
                "signal": "看多",
                "content": "均线多头排列，中长期趋势向上，可顺势做多",
                "action": "持股待涨或逢低买入"
            })
        elif "空头" in trend["ma_arrangement"]:
            suggestions.append({
                "type": "趋势",
                "signal": "看空",
                "content": "均线空头排列，中长期趋势向下，注意风险控制",
                "action": "减仓或观望"
            })
        else:
            suggestions.append({
                "type": "趋势",
                "signal": "中性",
                "content": "均线缠绕，处于震荡整理阶段",
                "action": "区间操作或观望"
            })

        # MACD 建议
        if "金叉" in macd["cross_signal"]:
            suggestions.append({
                "type": "MACD",
                "signal": "买入",
                "content": f"MACD金叉，{macd['zero_axis']}，{macd['macd_color']}",
                "action": "可考虑买入或加仓"
            })
        elif "死叉" in macd["cross_signal"]:
            suggestions.append({
                "type": "MACD",
                "signal": "卖出",
                "content": f"MACD死叉，{macd['zero_axis']}，{macd['macd_color']}",
                "action": "建议减仓或止损"
            })

        if macd["divergence"]["type"] != "无明显背离":
            suggestions.append({
                "type": "背离",
                "signal": "警示",
                "content": macd["divergence"]["type"] + " - " + macd["divergence"]["description"],
                "action": "重点关注，可能变盘"
            })

        # KDJ 建议
        if "超卖" in kdj["zone"]:
            suggestions.append({
                "type": "KDJ",
                "signal": "超卖",
                "content": f"KDJ进入超卖区(K={kdj['k']:.1f})，{kdj['zone_advice']}",
                "action": "关注反弹机会"
            })
        elif "超买" in kdj["zone"]:
            suggestions.append({
                "type": "KDJ",
                "signal": "超买",
                "content": f"KDJ进入超买区(K={kdj['k']:.1f})，{kdj['zone_advice']}",
                "action": "注意高位风险"
            })

        # 量能建议
        if "量价齐升" in volume["vol_price"]:
            suggestions.append({
                "type": "量能",
                "signal": "健康",
                "content": f"量价配合良好，{volume['vol_status']}",
                "action": "上涨有量能支撑"
            })
        elif "价跌量增" in volume["vol_price"]:
            suggestions.append({
                "type": "量能",
                "signal": "警示",
                "content": f"放量下跌，{volume['vol_status']}",
                "action": "注意风险，可能有恐慌盘"
            })

        # 布林带建议
        if "突破上轨" in boll["position"]:
            suggestions.append({
                "type": "布林",
                "signal": "超买",
                "content": f"价格突破布林带上轨，{boll['width_status']}",
                "action": "短期超买，注意回调"
            })
        elif "跌破下轨" in boll["position"]:
            suggestions.append({
                "type": "布林",
                "signal": "超卖",
                "content": f"价格跌破布林带下轨，{boll['width_status']}",
                "action": "短期超卖，关注反弹"
            })

        # K线形态建议
        latest_pattern = df.iloc[-1]["pattern"]
        if latest_pattern:
            pattern_type = df.iloc[-1]["pattern_type"]
            signal = "看多" if pattern_type == "bullish" else "看空" if pattern_type == "bearish" else "中性"
            suggestions.append({
                "type": "形态",
                "signal": signal,
                "content": f"出现{latest_pattern}形态",
                "action": "结合其他指标确认"
            })

        return suggestions

    def _generate_comprehensive_advice(self, df: pd.DataFrame) -> Dict[str, Any]:
        """生成综合操作建议（核心决策）"""
        latest = df.iloc[-1]
        score = self._calculate_score(df)
        trend = self._analyze_trend(df)
        macd = self._analyze_macd(df)
        kdj = self._analyze_kdj(df)
        volume = self._analyze_volume(df)
        boll = self._analyze_bollinger(df)

        # 信号统计
        bullish_signals = 0
        bearish_signals = 0
        signal_details = []

        # 1. 趋势信号
        if "多头" in trend["ma_arrangement"]:
            bullish_signals += 2
            signal_details.append(("趋势", "多", "均线多头排列"))
        elif "空头" in trend["ma_arrangement"]:
            bearish_signals += 2
            signal_details.append(("趋势", "空", "均线空头排列"))
        else:
            signal_details.append(("趋势", "平", "均线缠绕震荡"))

        # 2. MACD信号
        if "金叉" in macd["cross_signal"]:
            bullish_signals += 2
            signal_details.append(("MACD", "多", macd["cross_signal"]))
        elif "死叉" in macd["cross_signal"]:
            bearish_signals += 2
            signal_details.append(("MACD", "空", macd["cross_signal"]))
        else:
            if macd["cross_type"] == "bullish":
                bullish_signals += 1
                signal_details.append(("MACD", "多", "多头运行"))
            else:
                bearish_signals += 1
                signal_details.append(("MACD", "空", "空头运行"))

        # 3. KDJ信号
        if "超卖" in kdj["zone"]:
            bullish_signals += 1
            signal_details.append(("KDJ", "多", f"超卖区K={kdj['k']:.1f}"))
        elif "超买" in kdj["zone"]:
            bearish_signals += 1
            signal_details.append(("KDJ", "空", f"超买区K={kdj['k']:.1f}"))
        else:
            signal_details.append(("KDJ", "平", f"正常区域K={kdj['k']:.1f}"))

        # 4. 量价信号
        if "量价齐升" in volume["vol_price"]:
            bullish_signals += 1
            signal_details.append(("量价", "多", "量价齐升"))
        elif "价跌量增" in volume["vol_price"]:
            bearish_signals += 1
            signal_details.append(("量价", "空", "放量下跌"))
        elif "价涨量缩" in volume["vol_price"]:
            signal_details.append(("量价", "警", "量价背离"))
        else:
            signal_details.append(("量价", "平", "量能平稳"))

        # 5. 布林带信号
        if "突破上轨" in boll["position"] or boll["position_pct"] > 90:
            bearish_signals += 1
            signal_details.append(("布林", "空", "接近或突破上轨"))
        elif "跌破下轨" in boll["position"] or boll["position_pct"] < 10:
            bullish_signals += 1
            signal_details.append(("布林", "多", "接近或跌破下轨"))
        else:
            signal_details.append(("布林", "平", f"中轨附近({boll['position_pct']:.0f}%位置)"))

        # 6. 背离信号
        if "顶背离" in macd["divergence"]["type"]:
            bearish_signals += 2
            signal_details.append(("背离", "空", "MACD顶背离"))
        elif "底背离" in macd["divergence"]["type"]:
            bullish_signals += 2
            signal_details.append(("背离", "多", "MACD底背离"))

        # 综合判断
        net_signal = bullish_signals - bearish_signals
        total_signals = bullish_signals + bearish_signals

        # 确定操作建议
        if net_signal >= 4:
            operation = "强烈买入"
            operation_detail = "多重看多信号共振，建议积极买入建仓"
            position_advice = "可用60-80%仓位"
            risk_level = "低"
        elif net_signal >= 2:
            operation = "买入"
            operation_detail = "多数指标看多，可考虑分批买入"
            position_advice = "建议40-60%仓位"
            risk_level = "中低"
        elif net_signal >= 1:
            operation = "轻仓买入"
            operation_detail = "信号偏多但不强烈，可轻仓试探"
            position_advice = "建议20-40%仓位"
            risk_level = "中"
        elif net_signal <= -4:
            operation = "强烈卖出"
            operation_detail = "多重看空信号共振，建议清仓或重仓做空"
            position_advice = "建议清仓观望"
            risk_level = "高"
        elif net_signal <= -2:
            operation = "卖出"
            operation_detail = "多数指标看空，建议减仓或离场"
            position_advice = "减仓至20%以下"
            risk_level = "中高"
        elif net_signal <= -1:
            operation = "减仓"
            operation_detail = "信号偏空，建议适当减仓控制风险"
            position_advice = "减仓至50%以下"
            risk_level = "中"
        else:
            operation = "观望"
            operation_detail = "多空信号交织，建议持币观望等待明确方向"
            position_advice = "维持现有仓位或空仓"
            risk_level = "中"

        # 计算止损止盈位
        current_price = latest["close"]
        atr = latest.get("atr", current_price * 0.02)
        support1 = latest.get("support1", current_price * 0.95)
        resistance1 = latest.get("resistance1", current_price * 1.05)

        # 止损位：取支撑位和2倍ATR的较小值
        stop_loss = max(support1 * 0.99, current_price - 2 * atr)
        stop_loss_pct = (stop_loss / current_price - 1) * 100

        # 止盈位：取压力位和3倍ATR的较小值
        take_profit = min(resistance1 * 1.01, current_price + 3 * atr)
        take_profit_pct = (take_profit / current_price - 1) * 100

        # 风险收益比
        risk = current_price - stop_loss
        reward = take_profit - current_price
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # 入场时机
        if net_signal > 0:
            entry_timing = self._get_bullish_entry_timing(df, trend, macd, kdj)
        elif net_signal < 0:
            entry_timing = self._get_bearish_entry_timing(df, trend, macd, kdj)
        else:
            entry_timing = "等待方向明确后再做决策"

        return {
            "operation": operation,
            "operation_detail": operation_detail,
            "position_advice": position_advice,
            "risk_level": risk_level,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "net_signal": net_signal,
            "signal_details": signal_details,
            "stop_loss": round(stop_loss, 2),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "take_profit": round(take_profit, 2),
            "take_profit_pct": round(take_profit_pct, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "entry_timing": entry_timing,
            "score": score["total"],
            "rating": score["rating"],
        }

    def _get_bullish_entry_timing(self, df: pd.DataFrame, trend: Dict, macd: Dict, kdj: Dict) -> str:
        """获取看多时的入场时机建议"""
        latest = df.iloc[-1]
        suggestions = []

        # 回调买入点
        ma10 = trend["ma_values"].get("ma10", 0)
        if trend["above_ma_count"] >= 3 and ma10 > 0:
            suggestions.append(f"回调至MA10({ma10:.2f})附近可加仓")

        # KDJ入场点
        if kdj["k"] < 30:
            suggestions.append("当前KDJ超卖，可直接入场")
        elif kdj["k"] < 50:
            suggestions.append("等待KDJ金叉确认后入场")
        else:
            suggestions.append("等待回调至KDJ中性区域再入场")

        # MACD入场点
        if "金叉" in macd["cross_signal"]:
            suggestions.append("MACD刚金叉，可积极入场")
        elif macd["cross_type"] == "bullish" and macd["macd"] < 0:
            suggestions.append("等待MACD柱由绿转红时入场")

        # 布林带入场点
        boll = self._analyze_bollinger(df)
        if boll["position_pct"] < 30:
            suggestions.append("价格接近布林下轨，可考虑入场")

        return "；".join(suggestions) if suggestions else "可择机分批入场"

    def _get_bearish_entry_timing(self, df: pd.DataFrame, trend: Dict, macd: Dict, kdj: Dict) -> str:
        """获取看空时的出场/做空时机建议"""
        latest = df.iloc[-1]
        suggestions = []

        # 反弹卖出点
        ma10 = trend["ma_values"].get("ma10", 0)
        if trend["above_ma_count"] <= 1 and ma10 > 0:
            suggestions.append(f"反弹至MA10({ma10:.2f})附近可减仓")

        # KDJ出场点
        if kdj["k"] > 70:
            suggestions.append("当前KDJ超买，建议立即减仓")
        elif kdj["k"] > 50:
            suggestions.append("等待KDJ死叉确认后出场")
        else:
            suggestions.append("已在弱势区域，逢反弹减仓")

        # MACD出场点
        if "死叉" in macd["cross_signal"]:
            suggestions.append("MACD刚死叉，应果断减仓")
        elif macd["cross_type"] == "bearish" and macd["macd"] > 0:
            suggestions.append("等待MACD柱由红转绿时出场")

        # 布林带出场点
        boll = self._analyze_bollinger(df)
        if boll["position_pct"] > 70:
            suggestions.append("价格接近布林上轨，可考虑减仓")

        return "；".join(suggestions) if suggestions else "建议逢高减仓或清仓"

    def _generate_professional_report(
        self, df: pd.DataFrame, symbol: str, stock_name: str, analysis_type: str
    ) -> str:
        """生成专业分析报告"""
        latest = df.iloc[-1]
        report = []

        # ==================== 报告头部 ====================
        report.append("╔" + "═" * 68 + "╗")
        report.append("║" + f"  📊 股票技术分析报告".center(60) + "║")
        report.append("╠" + "═" * 68 + "╣")
        report.append(f"║  股票代码: {symbol}  股票名称: {stock_name or 'N/A'}".ljust(69) + "║")
        report.append(f"║  报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(69) + "║")
        report.append(f"║  数据区间: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')} (共{len(df)}个交易日)".ljust(69) + "║")
        report.append("╚" + "═" * 68 + "╝")
        report.append("")

        # ==================== 综合评分 ====================
        score = self._calculate_score(df)
        report.append("┌" + "─" * 68 + "┐")
        report.append("│" + "  【综合评分】".ljust(62) + "│")
        report.append("├" + "─" * 68 + "┤")
        report.append(f"│  总分: {score['total']}/100  评级: {score['rating']}".ljust(62) + "│")
        report.append("│" + "─" * 68 + "│")
        score_bar = "│  "
        for name, s in score["scores"].items():
            score_bar += f"{name}:{s} "
        report.append(score_bar.ljust(69) + "│")
        report.append("└" + "─" * 68 + "┘")
        report.append("")

        # ==================== 基本行情 ====================
        report.append("┌" + "─" * 68 + "┐")
        report.append("│" + "  【基本行情数据】".ljust(62) + "│")
        report.append("├" + "─" * 68 + "┤")
        report.append(f"│  最新价: {latest['close']:.2f}    开盘价: {latest['open']:.2f}    最高价: {latest['high']:.2f}    最低价: {latest['low']:.2f}".ljust(62) + "│")
        report.append(f"│  涨跌额: {latest.get('change', 0):+.2f}    涨跌幅: {latest.get('pct_change', 0):+.2f}%    振幅: {latest.get('amplitude', 0):.2f}%".ljust(62) + "│")

        volume = self._analyze_volume(df)
        report.append(f"│  成交量: {volume['volume']:,}手    成交额: {volume['amount']:.2f}亿    换手率: {volume['turnover']:.2f}%".ljust(62) + "│")
        report.append("└" + "─" * 68 + "┘")
        report.append("")

        # ==================== 趋势分析 ====================
        if analysis_type in ["full", "trend", "ma"]:
            trend = self._analyze_trend(df)
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【趋势分析】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  均线排列: {trend['ma_arrangement']}".ljust(62) + "│")
            report.append(f"│  价格位置: 站上{trend['above_ma_count']}条均线，跌破{trend['below_ma_count']}条均线".ljust(62) + "│")
            report.append("│" + "─" * 68 + "│")
            report.append(f"│  1日涨跌: {trend['pct_1d']:+.2f}%    5日涨跌: {trend['pct_5d']:+.2f}%    20日涨跌: {trend['pct_20d']:+.2f}%".ljust(62) + "│")
            report.append(f"│  20日波动率: {trend['volatility_20d']:.2f}%    14日振幅: {trend['trend_range']:.2f}%".ljust(62) + "│")
            report.append("│" + "─" * 68 + "│")
            report.append("│  均线数据:".ljust(69) + "│")
            ma_line = "│    "
            for name, val in trend["ma_values"].items():
                if pd.notna(val) and val > 0:
                    ma_line += f"{name.upper()}: {val:.2f}  "
            report.append(ma_line.ljust(69) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== MACD 分析 ====================
        if analysis_type in ["full", "macd"]:
            macd = self._analyze_macd(df)
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【MACD 指标分析】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  DIF: {macd['dif']:.4f}    DEA: {macd['dea']:.4f}    MACD柱: {macd['macd']:.4f}".ljust(62) + "│")
            report.append(f"│  信号: {macd['cross_signal']}    位置: {macd['zero_axis']}".ljust(62) + "│")
            report.append(f"│  柱状图: {macd['macd_color']}".ljust(62) + "│")
            report.append(f"│  背离检测: {macd['divergence']['type']}".ljust(62) + "│")
            if macd['divergence']['description']:
                report.append(f"│    → {macd['divergence']['description']}".ljust(62) + "│")
            report.append("│" + "─" * 68 + "│")
            report.append("│  近5日MACD数据:".ljust(69) + "│")
            report.append("│    日期          DIF       DEA       MACD".ljust(69) + "│")
            for _, row in macd["history"].iterrows():
                report.append(f"│    {row['date'].strftime('%Y-%m-%d')}    {row['dif']:+.4f}   {row['dea']:+.4f}   {row['macd']:+.4f}".ljust(62) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== KDJ 分析 ====================
        if analysis_type in ["full", "kdj"]:
            kdj = self._analyze_kdj(df)
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【KDJ 指标分析】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  K值: {kdj['k']:.2f}    D值: {kdj['d']:.2f}    J值: {kdj['j']:.2f}".ljust(62) + "│")
            report.append(f"│  信号: {kdj['cross_signal']}    区域: {kdj['zone']}".ljust(62) + "│")
            report.append(f"│  建议: {kdj['zone_advice']}".ljust(62) + "│")
            if kdj['j_blunt']:
                report.append(f"│  钝化: {kdj['j_blunt']}".ljust(62) + "│")
            report.append("│" + "─" * 68 + "│")
            report.append("│  近5日KDJ数据:".ljust(69) + "│")
            report.append("│    日期          K值      D值      J值".ljust(69) + "│")
            for i in range(-5, 0):
                if abs(i) <= len(df):
                    row = df.iloc[i]
                    report.append(f"│    {row['date'].strftime('%Y-%m-%d')}    {row['k']:6.2f}   {row['d']:6.2f}   {row['j']:6.2f}".ljust(62) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== RSI 分析 ====================
        if analysis_type == "full":
            rsi = self._analyze_rsi(df)
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【RSI 指标分析】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  RSI6: {rsi['rsi6']:.2f}    RSI12: {rsi['rsi12']:.2f}    RSI24: {rsi['rsi24']:.2f}".ljust(62) + "│")
            report.append(f"│  区域: {rsi['zone']}    趋势: {rsi['trend']}    信号: {rsi['signal']}".ljust(62) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== 布林带分析 ====================
        if analysis_type == "full":
            boll = self._analyze_bollinger(df)
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【布林带分析】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  上轨: {boll['upper']:.2f}    中轨: {boll['mid']:.2f}    下轨: {boll['lower']:.2f}".ljust(62) + "│")
            report.append(f"│  带宽: {boll['width']:.2f}%    {boll['width_status']}".ljust(62) + "│")
            report.append(f"│  位置: {boll['position']}".ljust(62) + "│")
            report.append(f"│  信号: {boll['signal']}".ljust(62) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== 量能分析 ====================
        if analysis_type == "full":
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【量能分析】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  今日成交量: {volume['volume']:,}手    5日均量: {volume['vol_ma5']:,}手".ljust(62) + "│")
            report.append(f"│  量比: {volume['volume_ratio']:.2f}    状态: {volume['vol_status']}".ljust(62) + "│")
            report.append(f"│  量价关系: {volume['vol_price']}".ljust(62) + "│")
            report.append(f"│  资金信号: {volume['vol_signal']}".ljust(62) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== K线形态 ====================
        if analysis_type in ["full", "pattern"]:
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【K线形态识别】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            recent_patterns = df.tail(10)[["date", "pattern", "pattern_type", "open", "close", "high", "low"]]
            has_pattern = False
            for _, row in recent_patterns.iterrows():
                if row["pattern"]:
                    has_pattern = True
                    signal_icon = "↑" if row["pattern_type"] == "bullish" else "↓" if row["pattern_type"] == "bearish" else "≈"
                    report.append(f"│  {row['date'].strftime('%Y-%m-%d')} {signal_icon} {row['pattern']}".ljust(62) + "│")
                    report.append(f"│    开:{row['open']:.2f} 收:{row['close']:.2f} 高:{row['high']:.2f} 低:{row['low']:.2f}".ljust(62) + "│")
            if not has_pattern:
                report.append("│  近10日无明显K线形态".ljust(69) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== 支撑压力位 ====================
        if analysis_type == "full":
            report.append("┌" + "─" * 68 + "┐")
            report.append("│" + "  【支撑位与压力位】".ljust(62) + "│")
            report.append("├" + "─" * 68 + "┤")
            report.append(f"│  当前价格: {latest['close']:.2f}".ljust(62) + "│")
            report.append("│" + "─" * 68 + "│")
            report.append(f"│  压力位1 (20日高点): {latest.get('resistance1', 0):.2f}  距离: {((latest.get('resistance1', latest['close'])/latest['close'])-1)*100:+.2f}%".ljust(62) + "│")
            report.append(f"│  压力位2 (枢轴R1):   {latest.get('r1', 0):.2f}  距离: {((latest.get('r1', latest['close'])/latest['close'])-1)*100:+.2f}%".ljust(62) + "│")
            report.append("│" + "─" * 68 + "│")
            report.append(f"│  支撑位1 (20日低点): {latest.get('support1', 0):.2f}  距离: {((latest.get('support1', latest['close'])/latest['close'])-1)*100:+.2f}%".ljust(62) + "│")
            report.append(f"│  支撑位2 (枢轴S1):   {latest.get('s1', 0):.2f}  距离: {((latest.get('s1', latest['close'])/latest['close'])-1)*100:+.2f}%".ljust(62) + "│")
            report.append("└" + "─" * 68 + "┘")
            report.append("")

        # ==================== 投资建议 ====================
        report.append("┌" + "─" * 68 + "┐")
        report.append("│" + "  【多空信号分析】".ljust(62) + "│")
        report.append("├" + "─" * 68 + "┤")
        suggestions = self._generate_suggestions(df)
        for i, s in enumerate(suggestions, 1):
            signal_color = "🟢" if s["signal"] in ["看多", "买入", "健康", "超卖"] else \
                          "🔴" if s["signal"] in ["看空", "卖出", "超买", "警示"] else "🟡"
            report.append(f"│  {signal_color} [{s['type']}] {s['content']}".ljust(60) + "│")
            report.append(f"│      → 操作建议: {s['action']}".ljust(62) + "│")
        report.append("└" + "─" * 68 + "┘")
        report.append("")

        # ==================== 综合操作建议（核心） ====================
        advice = self._generate_comprehensive_advice(df)
        report.append("╔" + "═" * 68 + "╗")
        report.append("║" + "  📋 综合操作建议".ljust(60) + "║")
        report.append("╠" + "═" * 68 + "╣")

        # 操作方向
        op_icon = "🟢" if "买" in advice["operation"] else "🔴" if "卖" in advice["operation"] or "减" in advice["operation"] else "🟡"
        report.append(f"║  {op_icon} 操作建议: {advice['operation']}".ljust(60) + "║")
        report.append(f"║  {advice['operation_detail']}".ljust(60) + "║")
        report.append("║" + "─" * 68 + "║")

        # 信号统计
        report.append(f"║  📊 信号统计: 看多{advice['bullish_signals']}个 vs 看空{advice['bearish_signals']}个 (净值:{advice['net_signal']:+d})".ljust(58) + "║")
        report.append("║" + "─" * 68 + "║")

        # 信号明细
        report.append("║  信号明细:".ljust(69) + "║")
        for sig_type, direction, detail in advice["signal_details"]:
            dir_icon = "🔼" if direction == "多" else "🔽" if direction == "空" else "➡️" if direction == "平" else "⚠️"
            report.append(f"║    {dir_icon} {sig_type}: {detail}".ljust(65) + "║")
        report.append("║" + "─" * 68 + "║")

        # 仓位与风险
        report.append(f"║  💰 仓位建议: {advice['position_advice']}".ljust(58) + "║")
        report.append(f"║  ⚡ 风险等级: {advice['risk_level']}".ljust(58) + "║")
        report.append("║" + "─" * 68 + "║")

        # 止损止盈
        report.append("║  🎯 关键价位:".ljust(69) + "║")
        report.append(f"║    止损价: {advice['stop_loss']:.2f} ({advice['stop_loss_pct']:+.2f}%)".ljust(65) + "║")
        report.append(f"║    止盈价: {advice['take_profit']:.2f} ({advice['take_profit_pct']:+.2f}%)".ljust(65) + "║")
        report.append(f"║    风险收益比: 1:{advice['risk_reward_ratio']:.1f}".ljust(65) + "║")
        report.append("║" + "─" * 68 + "║")

        # 入场时机
        report.append("║  ⏰ 入场/出场时机:".ljust(69) + "║")
        # 分行显示入场时机建议
        timing_parts = advice["entry_timing"].split("；")
        for part in timing_parts:
            if part.strip():
                report.append(f"║    • {part.strip()}".ljust(65) + "║")

        report.append("╚" + "═" * 68 + "╝")
        report.append("")

        # ==================== 风险提示 ====================
        # report.append("╔" + "═" * 68 + "╗")
        # report.append("║" + "  ⚠️ 风险提示".ljust(62) + "║")
        # report.append("╠" + "═" * 68 + "╣")
        # report.append("║  1. 以上分析基于历史数据，不代表未来走势".ljust(62) + "║")
        # report.append("║  2. 技术分析仅供参考，需结合基本面综合判断".ljust(62) + "║")
        # report.append("║  3. 投资有风险，入市需谨慎，请勿盲目跟从".ljust(62) + "║")
        # report.append("║  4. 建议设置止损位，控制仓位，分散投资".ljust(62) + "║")
        # report.append("╚" + "═" * 68 + "╝")

        return "\n".join(report)
