"""基金数据获取 Skill - 使用 AKShare 获取基金数据"""
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

from .base import BaseSkill, SkillOutput


class FundRetrieverSkill(BaseSkill):
    """基金数据获取 Skill - 获取基金净值和相关数据"""

    name = "fund_retriever"
    description = "获取基金历史净值数据、基金信息和业绩表现，返回JSON格式的分析数据"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fund_code": {
                    "type": "string",
                    "description": "基金代码，如 '008089' (华夏中证新能源汽车ETF联接A)"
                },
                "days": {
                    "type": "integer",
                    "description": "获取过去多少天的数据 (默认180天)"
                }
            },
            "required": ["fund_code"]
        }

    async def execute(
        self,
        fund_code: str,
        days: int = 180,
        **kwargs
    ) -> SkillOutput:
        # 处理 LangGraph 传入 None 的情况
        if days is None:
            days = 180

        try:
            # 1. 获取基金基本信息
            fund_info = await self._get_fund_info(fund_code)

            # 2. 获取基金净值数据
            df, fund_name = await self._fetch_fund_data(fund_code, days)
            if df is None or df.empty:
                return SkillOutput(
                    success=False,
                    result=None,
                    error=f"无法获取基金 {fund_code} 的数据"
                )

            # 3. 计算技术指标
            df = self._calculate_indicators(df)

            # 4. 生成分析数据JSON
            analysis_data = self._generate_analysis_json(df, fund_code, fund_name, fund_info)

            return SkillOutput(
                success=True,
                result=json.dumps(analysis_data, ensure_ascii=False, indent=2)
            )

        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"基金数据获取失败: {str(e)}"
            )

    async def _get_fund_info(self, fund_code: str) -> Dict[str, Any]:
        """获取基金基本信息"""
        import akshare as ak

        info = {
            "fund_code": fund_code,
            "fund_name": "",
            "fund_type": "",
            "fund_company": "",
            "establish_date": "",
            "fund_manager": "",
            "fund_size": "",
        }

        try:
            # 获取基金档案信息
            fund_info_df = ak.fund_individual_basic_info_xq(symbol=fund_code)
            if fund_info_df is not None and not fund_info_df.empty:
                for _, row in fund_info_df.iterrows():
                    item = row.get('item', '')
                    value = row.get('value', '')
                    if '基金全称' in item or '基金名称' in item:
                        info['fund_name'] = value
                    elif '基金类型' in item:
                        info['fund_type'] = value
                    elif '基金公司' in item or '管理人' in item:
                        info['fund_company'] = value
                    elif '成立日期' in item:
                        info['establish_date'] = value
                    elif '基金经理' in item:
                        info['fund_manager'] = value
                    elif '基金规模' in item:
                        info['fund_size'] = value
        except Exception as e:
            print(f"获取基金信息失败: {e}")

        return info

    async def _fetch_fund_data(
        self, fund_code: str, days: int
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """获取基金净值数据"""
        import akshare as ak

        fund_name = fund_code

        # 方法1: 开放式基金净值 (适用于大多数基金)
        try:
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "净值日期": "date",
                    "单位净值": "nav",  # Net Asset Value
                    "日增长率": "pct_change"
                })
                df["date"] = pd.to_datetime(df["date"])
                df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
                df["pct_change"] = pd.to_numeric(df["pct_change"], errors='coerce')
                df = df.sort_values("date").tail(days).reset_index(drop=True)

                # 尝试获取基金名称
                try:
                    name_df = ak.fund_name_em()
                    if name_df is not None:
                        match = name_df[name_df['基金代码'] == fund_code]
                        if not match.empty:
                            fund_name = match['基金简称'].values[0]
                except:
                    pass

                print(f"[开放式基金] 成功获取 {fund_name}({fund_code}) {len(df)} 条数据")
                return df, fund_name
        except Exception as e:
            print(f"[开放式基金] 获取失败: {e}")

        # 方法2: ETF基金历史数据
        try:
            df = ak.fund_etf_hist_em(symbol=fund_code, period="daily", adjust="qfq")
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "日期": "date",
                    "收盘": "nav",
                    "涨跌幅": "pct_change",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume"
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").tail(days).reset_index(drop=True)
                print(f"[ETF基金] 成功获取 {fund_code} {len(df)} 条数据")
                return df, fund_name
        except Exception as e:
            print(f"[ETF基金] 获取失败: {e}")

        return None, ""

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基金技术指标"""
        df = df.copy()

        # 确保 nav 列存在
        if 'nav' not in df.columns:
            return df

        # 均线系统
        for period in [5, 10, 20, 60]:
            df[f"ma{period}"] = df["nav"].rolling(window=period).mean()

        # 计算涨跌额
        df["change"] = df["nav"].diff()

        # 如果没有涨跌幅，计算它
        if "pct_change" not in df.columns or df["pct_change"].isna().all():
            df["pct_change"] = df["nav"].pct_change() * 100

        # 计算波动率
        df["volatility_20d"] = df["pct_change"].rolling(window=20).std()

        # 计算最大回撤
        df["cummax"] = df["nav"].cummax()
        df["drawdown"] = (df["nav"] - df["cummax"]) / df["cummax"] * 100

        # RSI (使用 Wilder's EMA)
        df = self._calculate_rsi(df)

        # 布林带
        df = self._calculate_bollinger(df)

        # MACD
        df = self._calculate_macd(df)

        # KDJ
        df = self._calculate_kdj(df)

        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 RSI 指标 (使用 Wilder's Smoothing / EMA)

        修复：原使用 SMA 计算，现改为 Wilder's EMA (alpha=1/period)
        这是 RSI 的标准计算方法，更加平滑且反应灵敏
        """
        if "pct_change" not in df.columns:
            return df

        delta = df["nav"].diff()
        # 使用 Wilder's smoothing: alpha = 1/period
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    def _calculate_bollinger(self, df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """计算布林带"""
        df["boll_mid"] = df["nav"].rolling(window=window).mean()
        std = df["nav"].rolling(window=window).std()
        df["boll_upper"] = df["boll_mid"] + num_std * std
        df["boll_lower"] = df["boll_mid"] - num_std * std
        # 添加布林带宽度指标
        df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"] * 100
        return df

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算 MACD 指标

        MACD (Moving Average Convergence Divergence) 是趋势跟踪动量指标
        - DIF: 快线与慢线的差值
        - DEA: DIF 的信号线 (EMA)
        - MACD: DIF 与 DEA 的差值 (柱状图)
        """
        df["ema12"] = df["nav"].ewm(span=fast, adjust=False).mean()
        df["ema26"] = df["nav"].ewm(span=slow, adjust=False).mean()
        df["dif"] = df["ema12"] - df["ema26"]
        df["dea"] = df["dif"].ewm(span=signal, adjust=False).mean()
        df["macd"] = (df["dif"] - df["dea"]) * 2

        # 判断金叉/死叉信号
        df["macd_signal"] = np.where(
            (df["dif"] > df["dea"]) & (df["dif"].shift(1) <= df["dea"].shift(1)), "金叉",
            np.where(
                (df["dif"] < df["dea"]) & (df["dif"].shift(1) >= df["dea"].shift(1)), "死叉",
                "无信号"
            )
        )
        return df

    def _calculate_kdj(self, df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
        """计算 KDJ 指标（基于净值模拟高低价）

        KDJ 是随机指标，用于判断超买超卖
        基金只有净值数据，使用滚动最大最小值模拟高低价
        """
        # 基金只有净值，用滚动最大最小值模拟
        low_min = df["nav"].rolling(window=n).min()
        high_max = df["nav"].rolling(window=n).max()
        rsv = (df["nav"] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)

        # 使用 EMA 平滑
        df["k"] = rsv.ewm(alpha=1/m1, adjust=False).mean()
        df["d"] = df["k"].ewm(alpha=1/m2, adjust=False).mean()
        df["j"] = 3 * df["k"] - 2 * df["d"]
        return df

    def _calculate_risk_metrics(self, df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """计算风险收益指标

        包括：夏普比率、卡玛比率、索提诺比率、胜率、盈亏比等
        """
        returns = df["pct_change"].dropna() / 100  # 转换为小数

        if len(returns) == 0:
            return {}

        # 年化收益率
        total_return = (df["nav"].iloc[-1] / df["nav"].iloc[0] - 1)
        trading_days = len(df)
        annual_return = (1 + total_return) ** (244 / trading_days) - 1 if trading_days > 0 else 0

        # 年化波动率（修正为 244 交易日，基金标准）
        annual_volatility = returns.std() * np.sqrt(244) if len(returns) > 1 else 0

        # 夏普比率 (Sharpe Ratio)
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        cummax = df["nav"].cummax()
        drawdown = (df["nav"] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 卡玛比率 (Calmar Ratio) - 年化收益 / 最大回撤
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 索提诺比率 (Sortino Ratio) - 只考虑下行波动
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(244) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

        # 胜率 (Win Rate) - 上涨天数占比
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

        # 盈亏比 (Profit/Loss Ratio) - 平均盈利 / 平均亏损
        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        profit_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0

        # 连涨/连跌天数
        streak = self._calculate_streak(returns)

        return {
            "annual_return": round(annual_return * 100, 2),
            "annual_volatility": round(annual_volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate": round(win_rate, 2),
            "profit_loss_ratio": round(profit_loss_ratio, 2),
            "current_streak": streak
        }

    def _calculate_streak(self, returns: pd.Series) -> Dict[str, Any]:
        """计算当前连涨/连跌天数"""
        if len(returns) == 0:
            return {"type": "无", "days": 0}

        current = returns.iloc[-1]
        streak_type = "涨" if current > 0 else ("跌" if current < 0 else "平")
        count = 1

        for i in range(len(returns) - 2, -1, -1):
            if (streak_type == "涨" and returns.iloc[i] > 0) or \
               (streak_type == "跌" and returns.iloc[i] < 0):
                count += 1
            else:
                break

        return {"type": streak_type, "days": count}

    def _get_kdj_zone(self, latest) -> str:
        """获取 KDJ 所在区域"""
        k = latest.get("k", 50)
        j = latest.get("j", 50)

        if pd.isna(k) or pd.isna(j):
            return "数据不足"

        if k > 80 or j > 100:
            return "超买区"
        elif k < 20 or j < 0:
            return "超卖区"
        elif k > 50:
            return "偏强区"
        elif k < 50:
            return "偏弱区"
        else:
            return "中性区"

    def _generate_analysis_json(
        self, df: pd.DataFrame, fund_code: str, fund_name: str, fund_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成分析数据的JSON结构"""
        latest = df.iloc[-1]

        # 计算区间收益
        returns = {}
        for days, label in [(5, "近5日"), (20, "近1月"), (60, "近3月"), (120, "近6月")]:
            if len(df) > days:
                start_nav = df.iloc[-days-1]["nav"] if len(df) > days else df.iloc[0]["nav"]
                end_nav = latest["nav"]
                returns[label] = round((end_nav / start_nav - 1) * 100, 2)
            else:
                returns[label] = None

        # 计算最大回撤
        max_drawdown = df["drawdown"].min() if "drawdown" in df.columns else 0

        # 计算年化波动率（修正为 244 交易日）
        if "pct_change" in df.columns:
            annual_volatility = df["pct_change"].std() * np.sqrt(244)
        else:
            annual_volatility = 0

        # 基本信息
        basic_info = {
            "fund_code": fund_code,
            "fund_name": fund_name or fund_info.get("fund_name", fund_code),
            "fund_type": fund_info.get("fund_type", ""),
            "fund_company": fund_info.get("fund_company", ""),
            "fund_manager": fund_info.get("fund_manager", ""),
            "fund_size": fund_info.get("fund_size", ""),
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_start": df['date'].iloc[0].strftime("%Y-%m-%d"),
            "data_end": df['date'].iloc[-1].strftime("%Y-%m-%d"),
            "trading_days": len(df)
        }

        # 最新净值
        latest_nav = {
            "nav": round(float(latest["nav"]), 4),
            "pct_change": round(float(latest.get("pct_change", 0)), 2),
            "change": round(float(latest.get("change", 0)), 4),
            "date": latest["date"].strftime("%Y-%m-%d")
        }

        # 均线数据
        ma_data = {}
        for period in [5, 10, 20, 60]:
            col = f"ma{period}"
            if col in df.columns and pd.notna(latest.get(col)):
                ma_data[col] = round(float(latest[col]), 4)
            else:
                ma_data[col] = None

        # 判断均线排列
        valid_mas = [v for v in ma_data.values() if v is not None]
        if len(valid_mas) >= 3:
            if valid_mas == sorted(valid_mas, reverse=True):
                ma_arrangement = "多头排列"
            elif valid_mas == sorted(valid_mas):
                ma_arrangement = "空头排列"
            else:
                ma_arrangement = "交叉缠绕"
        else:
            ma_arrangement = "数据不足"

        # 趋势数据
        trend_data = {
            "ma_values": ma_data,
            "ma_arrangement": ma_arrangement,
            "returns": returns,
            "max_drawdown": round(float(max_drawdown), 2),
            "annual_volatility": round(float(annual_volatility), 2),
            "volatility_20d": round(float(latest.get("volatility_20d", 0)), 2) if pd.notna(latest.get("volatility_20d")) else 0
        }

        # RSI数据
        rsi_data = {
            "rsi": round(float(latest.get("rsi", 50)), 2) if pd.notna(latest.get("rsi")) else 50,
            "zone": self._get_rsi_zone(latest.get("rsi", 50))
        }

        # MACD 数据
        macd_data = {
            "dif": round(float(latest.get("dif", 0)), 4) if pd.notna(latest.get("dif")) else 0,
            "dea": round(float(latest.get("dea", 0)), 4) if pd.notna(latest.get("dea")) else 0,
            "macd": round(float(latest.get("macd", 0)), 4) if pd.notna(latest.get("macd")) else 0,
            "signal": latest.get("macd_signal", "无信号"),
            "ema12": round(float(latest.get("ema12", 0)), 4) if pd.notna(latest.get("ema12")) else 0,
            "ema26": round(float(latest.get("ema26", 0)), 4) if pd.notna(latest.get("ema26")) else 0
        }

        # KDJ 数据
        kdj_data = {
            "k": round(float(latest.get("k", 50)), 2) if pd.notna(latest.get("k")) else 50,
            "d": round(float(latest.get("d", 50)), 2) if pd.notna(latest.get("d")) else 50,
            "j": round(float(latest.get("j", 50)), 2) if pd.notna(latest.get("j")) else 50,
            "zone": self._get_kdj_zone(latest)
        }

        # 布林带数据
        boll_data = {
            "upper": round(float(latest.get("boll_upper", 0)), 4) if pd.notna(latest.get("boll_upper")) else None,
            "mid": round(float(latest.get("boll_mid", 0)), 4) if pd.notna(latest.get("boll_mid")) else None,
            "lower": round(float(latest.get("boll_lower", 0)), 4) if pd.notna(latest.get("boll_lower")) else None,
            "width": round(float(latest.get("boll_width", 0)), 2) if pd.notna(latest.get("boll_width")) else None,
            "position": self._get_boll_position(latest)
        }

        # 风险收益指标
        risk_metrics = self._calculate_risk_metrics(df)

        # 历史净值（最近10天）
        history = []
        for _, row in df.tail(10).iterrows():
            history.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "nav": round(float(row["nav"]), 4),
                "pct_change": round(float(row.get("pct_change", 0)), 2)
            })

        return {
            "basic_info": basic_info,
            "latest_nav": latest_nav,
            "trend": trend_data,
            "rsi": rsi_data,
            "macd": macd_data,
            "kdj": kdj_data,
            "bollinger": boll_data,
            "risk_metrics": risk_metrics,
            "history": history
        }

    def _get_rsi_zone(self, rsi: float) -> str:
        """获取RSI所在区域"""
        if pd.isna(rsi):
            return "数据不足"
        if rsi > 80:
            return "超买区"
        elif rsi > 70:
            return "偏强区"
        elif rsi < 20:
            return "超卖区"
        elif rsi < 30:
            return "偏弱区"
        else:
            return "中性区"

    def _get_boll_position(self, latest) -> str:
        """获取布林带位置"""
        nav = latest.get("nav", 0)
        upper = latest.get("boll_upper", nav)
        mid = latest.get("boll_mid", nav)
        lower = latest.get("boll_lower", nav)

        if pd.isna(upper) or pd.isna(mid) or pd.isna(lower):
            return "数据不足"

        if nav > upper:
            return "突破上轨(超买)"
        elif nav > mid:
            pct = (nav - mid) / (upper - mid) * 100 if upper != mid else 50
            return f"中轨上方({pct:.0f}%)"
        elif nav > lower:
            pct = (nav - lower) / (mid - lower) * 100 if mid != lower else 50
            return f"中轨下方({pct:.0f}%)"
        else:
            return "跌破下轨(超卖)"
