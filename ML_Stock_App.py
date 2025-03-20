import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import time
from sklearn.linear_model import LinearRegression

# 🌟 APPLY CLEAN THEME
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .search-box { padding: 10px; border-radius: 5px; background: #1E293B; color: white; font-size: 16px; }
    .btn { padding: 8px 15px; background: #1E40AF; color: white; border-radius: 5px; font-size: 14px; transition: 0.3s; }
    .btn:hover { background: #3B82F6; transform: scale(1.1); }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ✅ Initialize session state properly
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

if "selected_timeframe" not in st.session_state:
    st.session_state["selected_timeframe"] = "1D"

# 📌 SEARCH BOX
st.markdown("<h3 style='color:white;'>🔍 Search a Stock</h3>", unsafe_allow_html=True)
search_stock = st.text_input(
    "", 
    value=st.session_state["search_input"], 
    key="search_input", 
    placeholder="Type stock symbol (e.g., TSLA, MSFT)..."
)

# 📌 TIMEFRAME SELECTION (UPDATES AUTOMATICALLY)
st.markdown("<h3 style='color:white;'>Select Timeframe</h3>", unsafe_allow_html=True)
timeframes = ["1D", "5D", "1M", "3M", "YTD", "1Y", "3Y", "5Y", "Max"]
selected_timeframe = st.radio(
    "", 
    timeframes, 
    horizontal=True, 
    key="selected_timeframe", 
    index=timeframes.index(st.session_state["selected_timeframe"])
)

# ✅ Auto-update timeframe in session state
st.session_state["selected_timeframe"] = selected_timeframe.lower()

# 📌 FETCH TOP 5 STOCKS
def get_top_stocks():
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_data = []
    for stock in top_stocks:
        ticker = yf.Ticker(stock)
        price = ticker.history(period="1d")["Close"].iloc[-1]
        change_pct = ticker.info.get("52WeekChange", 0)
        change_amt = price * change_pct

        stock_data.append({
            "symbol": stock,
            "name": ticker.info.get("shortName", stock),
            "price": f"${price:.2f}",
            "change": f"{change_amt:.2f} ({change_pct:.2%})",
            "change_class": "positive" if change_pct > 0 else "negative"
        })
    return stock_data

# 📌 CLICKABLE TOP STOCKS (UPDATES CHART AUTOMATICALLY)
st.markdown("<h3 style='color:white;'>Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(5)  # Align in a row

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]
            st.session_state["search_input"] = ""  # ✅ FIXED: Ensure it clears properly
            st.rerun()  # Refresh the app after selection

# ✅ USE SEARCH INPUT OR SELECTED STOCK
selected_stock = st.session_state["search_input"] if st.session_state["search_input"] else st.session_state["selected_stock"]

# 📌 PREDICT NEXT 30 DAYS
def predict_next_30_days(df):
    if df.empty or len(df) < 10:
        return None

    df["Days"] = np.arange(len(df))
    model = LinearRegression().fit(df[["Days"]], df["Close"])

    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_predictions = model.predict(future_days)

    return future_predictions

# 📌 LOAD STOCK DATA & DISPLAY GRAPH (UPDATED AUTOMATICALLY)
def plot_stock_chart(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    hist = ticker.history(period=st.session_state["selected_timeframe"])

    fig = go.Figure()

    # 🎯 STOCK PRICE LINE
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name=f"{stock_symbol} Close Price",
        line=dict(width=2)
    ))

    # 📌 PREDICT 30-DAY FUTURE PRICES
    future_predictions = predict_next_30_days(hist)
    if future_predictions is not None:
        future_dates = pd.date_range(start=hist.index[-1], periods=30, freq="D")
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode="lines",
            name=f"{stock_symbol} 30-Day Forecast",
            line=dict(dash="dash", color="orange")
        ))

    fig.update_layout(
        title=f"{stock_symbol} Stock Price & Forecast",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="white"),
        legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    return future_predictions

# 🎯 DISPLAY STOCK CHART (UPDATES WITH TIMEFRAME AND STOCK SELECTION)
future_predictions = plot_stock_chart(selected_stock)

# ✅ BUY/SELL RECOMMENDATION
st.markdown("<h3 style='color:white;'>📈 Recommendation</h3>", unsafe_allow_html=True)
if future_predictions is not None:
    last_close_price = yf.Ticker(selected_stock).history(period="1d")["Close"].iloc[-1]
    future_price = future_predictions[-1]

    if future_price > last_close_price:
        st.markdown("<h3 style='color:green;'>✅ Recommendation: Buy - Stock expected to increase.</h3>", unsafe_allow_html=True)
    elif future_price < last_close_price:
        st.markdown("<h3 style='color:red;'>❌ Recommendation: Sell - Stock expected to decline.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:gray;'>⚖️ Recommendation: Hold - No significant change expected.</h3>", unsafe_allow_html=True)

# ✅ REAL-TIME PRICE UPDATES
st.markdown("<h3 style='color:white;'>📊 Real-Time Price Updates</h3>", unsafe_allow_html=True)
ticker = yf.Ticker(selected_stock)
price_placeholder = st.empty()

while True:
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    price_placeholder.markdown(f"<h2 style='color:white;'>💲 {current_price:.2f}</h2>", unsafe_allow_html=True)
    time.sleep(30)  # Update every 30 seconds
