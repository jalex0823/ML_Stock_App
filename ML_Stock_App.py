import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# ---- Initialize Session ----
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"
if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# ---- Style ----
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-btn {
        width: 100%; height: 100px; font-size: 16px; text-align: center;
        background: #1E40AF; color: white; border-radius: 5px;
        border: none; transition: 0.3s; cursor: pointer; padding: 15px;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .stock-btn:hover { background: #3B82F6; transform: scale(1.05); }
    .info-box {
        font-size: 16px; padding: 10px; background: #1E293B;
        color: white; border-radius: 5px; margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Get S&P 500 ----
@st.cache_data
def get_sp500_list():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)[0]
        return table[['Security', 'Symbol']].dropna()
    except:
        return pd.DataFrame(columns=['Security', 'Symbol'])

sp500_list = get_sp500_list()

def get_stock_symbol(search_input):
    search_input = search_input.strip().upper()
    if search_input in sp500_list['Symbol'].values:
        return search_input
    result = process.extractOne(search_input, sp500_list['Security'])
    if result and result[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == result[0], 'Symbol'].values[0]
    return None

# ---- Dynamic Real-Time Top Stocks ----
def get_top_stocks():
    tickers = sp500_list['Symbol'].tolist()[:50]
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            price = info.get("regularMarketPrice", 0)
            change = info.get("52WeekChange", 0)
            delta = price * change
            data.append({
                "symbol": t,
                "name": info.get("shortName", t),
                "price": price,
                "change": delta,
                "percent": change
            })
        except:
            continue
    return sorted(data, key=lambda x: x["percent"], reverse=True)[:15]

# ---- UI Elements ----
st.markdown("<h3 style='color:white;'>🔍 Search by Company Name or Symbol</h3>", unsafe_allow_html=True)
search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or company name...").strip().upper()

st.markdown("<h3 style='color:white;'>📈 Top 15 Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
col1, col2, col3 = st.columns(3)

for i, stock in enumerate(top_stocks):
    col = [col1, col2, col3][i % 3]
    with col:
        label = f"**{stock['name']}**\n{stock['symbol']}\n💲{stock['price']:.2f}\n📈 {stock['change']:+.2f} ({stock['percent']:.2%})"
        if st.button(label, key=f"top_{i}", use_container_width=True):
            st.session_state["search_input"] = ""
            st.session_state["selected_stock"] = stock["symbol"]

selected_stock = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]
if not selected_stock:
    st.error("⚠️ Invalid company name or symbol. Please try again.")
    st.stop()

# ---- Data Utilities ----
def get_stock_data(symbol):
    try:
        return yf.Ticker(symbol).history(period="1y")
    except:
        return None

def predict_next_30_days(df):
    if df is None or df.empty or len(df) < 30:
        return np.array([])
    df["Days"] = np.arange(len(df))
    model = LinearRegression().fit(df[["Days"]], df["Close"])
    return model.predict(np.arange(len(df), len(df)+30).reshape(-1, 1))

def get_recommendation(df):
    forecast = predict_next_30_days(df)
    if df is None or df.empty or forecast.size == 0:
        return "No data available"
    return "✅ Buy - Expected to Increase" if forecast[-1] > df["Close"].iloc[-1] else "❌ Sell - Expected to Decrease"

# ---- Chart ----
def plot_stock_chart(symbol):
    df = get_stock_data(symbol)
    if df is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{symbol} Close Price", line=dict(width=2)))
    ma_20 = df["Close"].rolling(window=20).mean()
    ma_500 = df["Close"].rolling(window=500).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma_20, mode="lines", name="20-Day MA", line=dict(color="blue", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=ma_500, mode="lines", name="500-Day MA", line=dict(color="red", dash="dot")))
    forecast = predict_next_30_days(df)
    if forecast.size > 0:
        future_dates = pd.date_range(start=df.index[-1], periods=30, freq="D")
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines", name="30-Day Forecast", line=dict(dash="dash", color="orange")))
    fig.update_layout(
        title=f"{symbol} Stock Price & Trends",
        xaxis_title="Date", yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font=dict(color="white"),
        legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Display Chart & Forecast ----
plot_stock_chart(selected_stock)
df = get_stock_data(selected_stock)
forecast = predict_next_30_days(df)
highest_forecast = np.max(forecast) if forecast.size > 0 else None
current_price = df["Close"].iloc[-1] if df is not None and not df.empty else None

if current_price is not None:
    st.markdown(f"<div class='info-box'>💲 Live Price: {current_price:.4f}</div>", unsafe_allow_html=True)
if highest_forecast:
    st.markdown(f"<div class='info-box'>📈 Highest Predicted Price (Next 30 Days): {highest_forecast:.4f}</div>", unsafe_allow_html=True)
    price_diff = highest_forecast - current_price
    symbol = "▲" if price_diff > 0 else "▼"
    st.markdown(f"<div class='info-box'>📉 Forecasted Change: {symbol} {price_diff:.2f}</div>", unsafe_allow_html=True)

st.markdown(f"<div class='info-box'>📊 Recommendation: {get_recommendation(df)}</div>", unsafe_allow_html=True)