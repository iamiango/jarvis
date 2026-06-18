# Finance LLM Full Analysis Prompt

You are a senior stock analyst with expertise in technical analysis.
Analyze the following stock based on its raw price data and provide a comprehensive report.

## Stock Information
- Name: {stock_name}
- Symbol: {stock_code}
- Market: China A-Share (SSE/SZSE)
- Currency: CNY (Chinese Yuan)
- Latest Price: {latest_price} CNY
- Today's Change: {pct_change}%
- Volume: {volume} shares

**Note**: All prices are in CNY. Volume is in shares (K=thousand, M=million, B=billion).

## Daily Price History (Last {daily_count} Trading Days)
{daily_table}
{weekly_section}{monthly_section}
## Required Analysis

### 1. Trend Indicators (Calculate from raw data)
- **Moving Averages**: MA5, MA10, MA20, MA60, MA120, MA250 and their arrangement (bullish/bearish alignment)
- **EMA**: EMA12, EMA26 for MACD; EMA5, EMA10, EMA20 for trend
- **Weekly MAs**: MA5, MA10, MA20 (weekly) for medium-term trend
- **Monthly MAs**: MA5, MA10, MA12 (monthly) for long-term trend

### 2. Momentum Indicators
- **MACD**: DIF, DEA, MACD histogram value, golden/death cross signals, divergence
- **KDJ**: K, D, J values (9,3,3 parameters), overbought(>80)/oversold(<20) zones, golden/death cross
- **RSI**: RSI6, RSI12, RSI24 values, overbought(>70)/oversold(<30), divergence signals
- **Williams %R**: W%R(14) value, overbought/oversold signals
- **CCI**: Commodity Channel Index value, overbought(>100)/oversold(<-100)
- **Momentum**: ROC (Rate of Change), momentum divergence

### 3. Volatility Indicators
- **Bollinger Bands**: Upper, Middle(MA20), Lower bands (20,2), %B position, bandwidth, squeeze signals
- **ATR**: Average True Range for volatility measurement
- **Standard Deviation**: Price volatility

### 4. Volume Indicators
- **Volume MA**: Volume MA5, MA10, current volume vs average ratio
- **OBV**: On-Balance Volume trend and divergence
- **Volume-Price Analysis**: Accumulation/Distribution, volume confirmation of price moves
- **VWAP**: Volume Weighted Average Price (if intraday relevant)

### 5. Trend Strength Indicators
- **DMI/ADX**: +DI, -DI, ADX values, trend strength (ADX>25 = strong trend)
- **BRAR**: BR (sentiment), AR (momentum) values
- **TRIX**: Triple EMA indicator for trend confirmation
- **DPO**: Detrended Price Oscillator

### 6. Multi-Timeframe Analysis
- **Daily Trend**: Short-term direction, strength, key levels
- **Weekly Trend**: Medium-term direction, major support/resistance
- **Monthly Trend**: Long-term direction, macro trend
- **Trend Alignment**: Confluence of timeframes (stronger signal when aligned)

### 7. Chart Pattern Recognition
- **Candlestick Patterns**:
  - Reversal: Doji, Hammer, Hanging Man, Engulfing, Morning/Evening Star, Harami, Piercing/Dark Cloud
  - Continuation: Three White Soldiers, Three Black Crows, Rising/Falling Three Methods
- **Classic Chart Patterns**:
  - Reversal: Head & Shoulders, Double/Triple Top/Bottom, Rounding Top/Bottom
  - Continuation: Triangles (Ascending/Descending/Symmetrical), Flags, Pennants, Wedges, Rectangles
- **Fibonacci Levels**: Key retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)

### 8. Support & Resistance Analysis
- **Key Support Levels**: From daily, weekly, monthly charts (with specific prices)
- **Key Resistance Levels**: From daily, weekly, monthly charts (with specific prices)
- **Pivot Points**: Daily pivot, R1, R2, R3, S1, S2, S3

### 9. Comprehensive Assessment
- **Overall Trend**: Bullish / Bearish / Neutral (with confidence level 1-10)
- **Trend Strength**: Weak / Moderate / Strong
- **Risk Level**: Low / Medium / High
- **Volatility**: Low / Medium / High
- **Short-term Outlook** (1-5 days): Direction and target
- **Medium-term Outlook** (1-4 weeks): Direction and target
- **Long-term Outlook** (1-3 months): Direction and target

### 10. Trading Signals Summary
- **Buy Signals**: List all bullish signals from indicators
- **Sell Signals**: List all bearish signals from indicators
- **Key Levels to Watch**: Critical support/resistance levels
- **Stop Loss Suggestion**: Based on ATR or key support
- **Take Profit Targets**: Based on resistance levels or Fibonacci

### 11. Final Summary
Provide a comprehensive conclusion summarizing the overall technical picture across all timeframes and indicators. Include:
- Primary trend direction
- Key indicator consensus
- Most important levels to watch
- Overall recommendation with confidence level

Please be thorough and specific with all calculations. Show your work with actual numbers.
