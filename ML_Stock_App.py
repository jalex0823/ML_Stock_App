import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# Initialize session state
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# UI Theme and styles
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-card {
        background: #1E293B;
        padding: 10px;
        margin: 8px;
        border-radius: 10px;
        color: white;
        text-align: center;
        transition: 0.3s;
        width: 100%;
    }
    .stock-card:hover {
        background: #334155;
        transform: scale(1.02);
        cursor: pointer;
    }
    .info-box {
        font-size: 16px;
        padding: 10px;
        background: #1E293B;
        color: white;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Fetch S&P 500 list
@st.cache_data
def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
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

def get_top_stocks():
    tickers = sp500_list['Symbol'].tolist()
    top_data = []
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            if hist.empty:
                continue
            price_now = hist["Close"].iloc[-1]
            price_before = hist["Close"].iloc[0]
            pct_change = (price_now - price_before) / price_before
            change_amt = price_now - price_before
            top_data.append({
                "symbol": symbol,
                "name": ticker.info.get("shortName", symbol),
                "price": price_now,
                "change": change_amt,
                "percent": pct_change
            })
        except:
            continue
    sorted_top = sorted(top_data, key=lambda x: x["percent"], reverse=True)
    return sorted_top[:15]

def get_stock_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1y")
        return data if not data.empty else None
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
    if forecast[-1] > df["Close"].iloc[-1]:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decrease"

# Search UI
st.markdown("<h3 style='color:white;'>🔍 Search by Company Name or Symbol</h3>", unsafe_allow_html=True)
search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or company name...").strip().upper()

# Display top performing stocks
st.markdown("<h3 style='color:white;'>📈 Top 15 Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(3)
for i, stock in enumerate(top_stocks):
    with cols[i % 3]:
        if st.button(f"{stock['name']} ({stock['symbol']})\n${stock['price']:.2f}\n{stock['change']:+.2f} ({stock['percent']:.2%})", key=f"top_{i}"):
            st.session_state["search_input"] = ""
            st.session_state["selected_stock"] = stock["symbol"]

selected_stock = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]
if not selected_stock:
    st.error("⚠️ Invalid company name or symbol. Please try again.")
    st.stop()

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
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="white"),
        legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# Display stock chart
plot_stock_chart(selected_stock)

# Price & Forecast Info
df = get_stock_data(selected_stock)
forecast = predict_next_30_days(df)
highest_forecast = np.max(forecast) if forecast.size > 0 else None
current_price = df["Close"].iloc[-1] if df is not None and not df.empty else None

if current_price is not None:
    st.markdown(f"<div class='info-box'>💲 Live Price: {current_price:.4f}</div>", unsafe_allow_html=True)

if highest_forecast:
    st.markdown(f"<div class='info-box'>📈 Highest Predicted Price (Next 30 Days): {highest_forecast:.4f}</div>", unsafe_allow_html=True)
    diff = highest_forecast - current_price
    sign = "+" if diff >= 0 else "-"
    st.markdown(f"<div class='info-box'>💡 Projected Gain/Loss: {sign}${abs(diff):.4f}</div>", unsafe_allow_html=True)

recommendation = get_recommendation(df)
st.markdown(f"<div class='info-box'>📊 Recommendation: {recommendation}</div>", unsafe_allow_html=True)