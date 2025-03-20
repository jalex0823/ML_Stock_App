import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import time

# 🌟 APPLY CLEAN THEME
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-card { padding: 15px; margin: 5px; background: linear-gradient(135deg, #1E293B, #334155); 
                  border-radius: 10px; color: white; text-align: center; transition: 0.3s; cursor: pointer; }
    .stock-card:hover { transform: scale(1.05); background: linear-gradient(135deg, #334155, #475569); }
    .search-box { padding: 10px; border-radius: 5px; background: #1E293B; color: white; font-size: 16px; }
    .btn { padding: 8px 15px; background: #1E40AF; color: white; border-radius: 5px; font-size: 14px; transition: 0.3s; }
    .btn:hover { background: #3B82F6; transform: scale(1.1); }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 📌 FETCH TOP STOCKS (Fixing Yahoo Finance API issues)
def get_top_stocks():
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_data = []
    
    for stock in top_stocks:
        ticker = yf.Ticker(stock)
        try:
            price = ticker.fast_info["last_price"]
            change_pct = ticker.fast_info.get("52_week_change", 0)  # Use safe get method
            change_amt = price * change_pct

            stock_data.append({
                "symbol": stock,
                "name": stock,
                "price": f"${price:.2f}",
                "change": f"{change_amt:.2f} ({change_pct:.2%})",
                "change_class": "positive" if change_pct > 0 else "negative"
            })
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
            stock_data.append({
                "symbol": stock,
                "name": stock,
                "price": "N/A",
                "change": "N/A",
                "change_class": "neutral"
            })
    
    return stock_data

# ✅ INITIALIZE SESSION STATE
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

# 📌 SEARCH FEATURE
st.markdown("<h3 style='color:white;'>🔍 Search a Stock</h3>", unsafe_allow_html=True)
search_stock = st.text_input("", key="search_input", placeholder="Type stock symbol (e.g., TSLA, MSFT)...")

# 📌 DISPLAY TOP STOCKS
st.markdown("<h3 style='color:white;'>📌 Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(5)

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]
            st.session_state["search_input"] = ""  # ✅ Fix: Clear search bar when a top stock is selected

# ✅ USE SEARCH OR DEFAULT STOCK
selected_stock = search_stock if search_stock else st.session_state["selected_stock"]

# 📌 STOCK PREDICTION (30-Day Forecast)
import numpy as np
from sklearn.linear_model import LinearRegression

def predict_next_30_days(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    hist = ticker.history(period="1y")
    
    if hist.empty:
        return np.array([])

    hist['Days'] = np.arange(len(hist))
    X = hist[['Days']]
    y = hist['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = np.arange(len(hist), len(hist) + 30).reshape(-1, 1)
    return model.predict(future_days)

# 📌 DISPLAY STOCK CHART
def plot_stock_chart(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    hist = ticker.history(period="1y")  # ✅ Defaults to 1 year

    fig = go.Figure()

    # 🎯 Stock Price Line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name=f"{stock_symbol} Close Price",
        line=dict(width=2)
    ))

    # 📌 30-Day Forecast
    future_predictions = predict_next_30_days(stock_symbol)
    future_dates = pd.date_range(start=hist.index[-1], periods=30, freq='D')

    if future_predictions.size > 0:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode="lines",
            name="30-Day Forecast",
            line=dict(dash="dash", color="orange")
        ))

    fig.update_layout(
        title=f"{stock_symbol} Stock Price & Trends",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="white"),
        legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# 🎯 DISPLAY STOCK CHART
plot_stock_chart(selected_stock)

# 📌 BUY/SELL RECOMMENDATION
st.markdown("<h3 style='color:white;'>📢 Recommendation</h3>", unsafe_allow_html=True)
last_close_price = yf.Ticker(selected_stock).history(period="1d")["Close"].iloc[-1]
forecast_price = predict_next_30_days(selected_stock)[-1] if predict_next_30_days(selected_stock).size > 0 else last_close_price

if forecast_price > last_close_price:
    st.markdown(f"<h3 style='color:#16A34A;'>✅ Buy: Price expected to increase to ${forecast_price:.2f}</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h3 style='color:#DC2626;'>🚨 Sell: Price expected to drop to ${forecast_price:.2f}</h3>", unsafe_allow_html=True)

# 📌 REAL-TIME STOCK PRICE UPDATES
st.markdown("<h3 style='color:white;'>📊 Real-Time Price Updates</h3>", unsafe_allow_html=True)
ticker = yf.Ticker(selected_stock)
price_placeholder = st.empty()

while True:
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    price_placeholder.markdown(f"<h2 style='color:white;'>💲 {current_price:.2f}</h2>", unsafe_allow_html=True)
    time.sleep(30)  # Update every 30 seconds
