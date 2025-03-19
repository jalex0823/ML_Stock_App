import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import time

# 🎨 Custom CSS for UI Improvements
st.markdown("""
    <style>
    body { background-color: #0F172A; }
    .btn-group { display: flex; justify-content: center; margin-bottom: 10px; flex-wrap: wrap; }
    .btn { padding: 8px 15px; border: none; cursor: pointer; background: #1E40AF; color: white; border-radius: 5px; font-size: 14px; margin: 5px; }
    .btn:hover { background: #3B82F6; }
    .watchlist-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    .watchlist-table th, .watchlist-table td { padding: 10px; text-align: left; }
    .watchlist-table th { background-color: #1E293B; color: white; font-size: 14px; }
    .watchlist-table td { background-color: #0F172A; color: white; font-size: 13px; }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 📌 Timeframe Buttons
timeframe = st.selectbox("", ["1D", "5D", "1M", "3M", "YTD", "1Y", "5Y", "Max"], index=4, key="time_select")

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

# 📌 Display Stock List (Side-by-side with Graph)
st.markdown("<h3 style='color:white;'>Quick Compare</h3>", unsafe_allow_html=True)
st.markdown("<table class='watchlist-table'><tr><th>Stock</th><th>Trend</th><th>Price</th><th>Change</th></tr>", unsafe_allow_html=True)

for stock in stock_data:
    trend_fig = go.Figure(go.Scatter(y=stock["trend"], mode="lines", line=dict(color="white", width=1.5)))
    st.markdown(f"""
        <tr>
            <td><button class='btn' onclick="document.getElementById('stock_input').value='{stock['symbol']}';">{stock['name']} ({stock['symbol']})</button></td>
            <td>{st.plotly_chart(trend_fig, use_container_width=True)}</td>
            <td>{stock['price']}</td>
            <td class="{stock['change_class']}">{stock['change']}</td>
        </tr>
    """, unsafe_allow_html=True)

st.markdown("</table>", unsafe_allow_html=True)

# 🔍 Search Stock Input
search_stock = st.text_input("Search Stock:", key="stock_input")

# 📌 Stock Comparison Selection
st.markdown("<h3 style='color:white;'>Compare Stocks</h3>", unsafe_allow_html=True)
compare_stocks = st.multiselect("Select Stocks to Compare:", top_stocks)

# ⏳ Auto-refresh Interval (Real-Time Updates)
refresh_interval = st.slider("Update Interval (seconds)", min_value=5, max_value=60, value=30)

# 📈 Stock Graph with Comparison
if search_stock or compare_stocks:
    fig = go.Figure()

    # 📌 Fetch Data for Selected Stocks
    selected_stocks = [search_stock] if search_stock else []
    selected_stocks += compare_stocks

    for stock in selected_stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1y")

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
            model = LinearRegression()
            model.fit(X, y)
            future_days = np.arange(len(hist), len(hist) + 30).reshape(-1, 1)
            future_pred = model.predict(future_days)

            future_dates = pd.date_range(start=hist.index[-1], periods=30, freq="D")
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                mode="lines",
                name=f"{stock} 30-Day Forecast",
                line=dict(dash="dot")
            ))

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

    # 🏦 Stock Details
    for stock in selected_stocks:
        ticker = yf.Ticker(stock)
        stock_info = ticker.info
        st.markdown(f"<h3 style='color:white;'>Stock Details for {stock.upper()}</h3>", unsafe_allow_html=True)
        st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
        st.write(f"**Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
        st.write(f"**Share Price:** {format_currency(stock_info.get('regularMarketPrice', 0))}")
        st.write(f"**Yearly Change:** {format_currency(stock_info.get('52WeekChange', 0))}")

        # 📢 Recommendation
        if future_pred[-1] > hist["Close"].iloc[-1]:
            st.markdown("<p style='color:#16A34A; font-size:20px;'><b>✅ Recommendation: BUY - Stock expected to increase.</b></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#DC2626; font-size:20px;'><b>❌ Recommendation: SELL - Stock expected to decrease.</b></p>", unsafe_allow_html=True)

# 🔄 Auto Refresh
while True:
    time.sleep(refresh_interval)
    st.rerun()

