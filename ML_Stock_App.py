import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# 🎨 UI Styling
st.markdown("""
    <style>
    body { background-color: #0F172A; }
    .btn-group { display: flex; justify-content: center; margin-bottom: 10px; flex-wrap: wrap; }
    .btn { padding: 8px 15px; border: none; cursor: pointer; background: #1E40AF; color: white; border-radius: 5px; font-size: 14px; margin: 5px; transition: 0.3s; }
    .btn:hover { background: #3B82F6; transform: scale(1.05); }
    .table-container { margin-top: 20px; }
    .watchlist-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    .watchlist-table th, .watchlist-table td { padding: 10px; text-align: left; border: 1px solid #334155; }
    .watchlist-table th { background-color: #1E293B; color: white; font-size: 14px; }
    .watchlist-table td { background-color: #0F172A; color: white; font-size: 13px; }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 📌 Timeframe Selection (Using Buttons Instead of Dropdown)
st.markdown("<h3 style='color:white; text-align:center;'>Select Timeframe</h3>", unsafe_allow_html=True)
timeframes = ["1D", "5D", "1M", "3M", "YTD", "1Y", "5Y", "Max"]
selected_timeframe = st.radio("", timeframes, horizontal=True, key="time_select")

# 📊 Fetch Top 5 Stocks Data
top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
stock_data = []
for stock in top_stocks:
    ticker = yf.Ticker(stock)
    hist = ticker.history(period="6mo")
    price = ticker.info.get("regularMarketPrice", 0)
    change_pct = ticker.info.get("52WeekChange", 0)
    change_amt = price * change_pct

    stock_data.append({
        "symbol": stock,
        "name": ticker.info.get("shortName", stock),
        "price": f"${price:.2f}",
        "change": f"{change_amt:.2f} ({change_pct:.2%})",
        "change_class": "positive" if change_pct > 0 else "negative",
        "trend": hist["Close"][-10:].tolist() if not hist.empty else []
    })

# 📌 Layout Fix - Proper Positioning
col1, col2 = st.columns([1, 3], gap="large")  # Adjusted for better spacing

# 🟢 **LEFT COLUMN: Quick Compare Stocks**
with col1:
    st.markdown("<h3 style='color:white;'>Quick Compare</h3>", unsafe_allow_html=True)
    
    # 📌 Properly formatted stock comparison table
    table_html = "<div class='table-container'><table class='watchlist-table'><tr><th>Stock</th><th>Price</th><th>Change</th></tr>"

    for stock in stock_data:
        table_html += f"""
            <tr>
                <td><button class='btn' onclick="document.getElementById('stock_input').value='{stock['symbol']}';">{stock['name']} ({stock['symbol']})</button></td>
                <td>{stock['price']}</td>
                <td class="{stock['change_class']}">{stock['change']}</td>
            </tr>
        """

    table_html += "</table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

# 🔍 **SEARCH STOCK INPUT BELOW TABLE**
search_stock = st.text_input("Search Stock:", key="stock_input")

# 📌 Stock Comparison Selection (Handling Empty Selections Properly)
st.markdown("<h3 style='color:white;'>Compare Stocks</h3>", unsafe_allow_html=True)
compare_stocks = st.multiselect("Select Stocks to Compare:", top_stocks)

# ✅ Fix for NameError by Ensuring `selected_stocks` is Always Defined
selected_stocks = []
if search_stock:
    selected_stocks.append(search_stock)
selected_stocks += compare_stocks

# 🟢 **RIGHT COLUMN: Stock Graph**
with col2:
    if selected_stocks:
        fig = go.Figure()

        # 📌 Fetch Data for Selected Stocks
        for stock in selected_stocks:
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period=selected_timeframe.lower())

                if not hist.empty:
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist["Close"],
                        mode="lines",
                        name=f"{stock} Close Price",
                        line=dict(width=2)
                    ))

                    # 📊 Moving Averages
                    ma_50 = hist["Close"].rolling(window=50).mean()
                    ma_200 = hist["Close"].rolling(window=200).mean()
                    show_50_ma = st.checkbox(f"Show 50-Day MA for {stock}", value=True)
                    show_200_ma = st.checkbox(f"Show 200-Day MA for {stock}", value=False)

                    if show_50_ma:
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=ma_50,
                            mode="lines",
                            name=f"{stock} 50-Day MA",
                            line=dict(dash="dash")
                        ))

                    if show_200_ma:
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=ma_200,
                            mode="lines",
                            name=f"{stock} 200-Day MA",
                            line=dict(dash="dash")
                        ))

                    # 📈 Forecast for Next 30 Days
                    hist["Days"] = np.arange(len(hist))
                    X = hist[["Days"]]
                    y = hist["Close"]
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    future_days = np.arange(len(hist), len(hist) + 30).reshape(-1, 1)
                    future_pred = model.predict(future_days)

                    future_dates = pd.date_range(start=hist.index[-1], periods=30, freq="D")
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_pred,
                        mode="lines",
                        name=f"{stock} 30-Day Forecast",
                        line=dict(dash="dot", color="yellow")
                    ))

            except Exception as e:
                st.warning(f"Could not retrieve data for {stock}: {e}")

        # 📌 Format Chart
        fig.update_layout(
            title="Stock Price & Trends",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="white"),
            legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

# ✅ Fix for NameError: Ensuring Stock Selection is Always Valid
if not selected_stocks:
    st.info("Select or search for a stock to display its details.")
