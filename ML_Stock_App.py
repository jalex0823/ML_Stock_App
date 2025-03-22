import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process
import time

# Must be first
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

# CSS + Title
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
    h1 { text-align: center; color: white; font-size: 36px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>THE AI STOCK ANALYZER AND PREDICTIONS APP</h1>", unsafe_allow_html=True)

# Cache stock list
@st.cache_data
def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    return table[['Security', 'Symbol']].dropna()

sp500_list = get_sp500_list()

# Symbol fetch
def get_stock_symbol(query):
    query = query.strip().upper()
    if query in sp500_list['Symbol'].values:
        return query
    match = process.extractOne(query, sp500_list['Security'])
    if match and match[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == match[0], 'Symbol'].values[0]
    return None

# Faster top 15 stocks
@st.cache_data(ttl=30)
def get_top_15_fast():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
        "BRK-B", "V", "JPM", "UNH", "MA", "HD", "PG", "LLY", "PEP", "BAC",
        "COST", "AVGO", "ABBV", "WMT", "KO", "MRK", "XOM"
    ]
    data = []
    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            info = stock.fast_info
            price = info.get("last_price", 0)
            prev = info.get("previous_close", 0)
            if price and prev:
                change_pct = (price - prev) / prev
                data.append({
                    "symbol": symbol,
                    "price": price,
                    "change_pct": change_pct
                })
        except:
            continue
    df = pd.DataFrame(data)
    df = df.sort_values(by="change_pct", ascending=False).head(15).reset_index(drop=True)
    df["price"] = df["price"].round(2)
    df["change_pct"] = (df["change_pct"] * 100).round(2)
    return df

# Search + buttons
search_input = st.text_input("🔍 Search by Company Name or Symbol", value=st.session_state.get("search_input", "")).upper()
top_15 = get_top_15_fast()

col1, col2, col3 = st.columns(3)
for i, row in top_15.iterrows():
    col = [col1, col2, col3][i % 3]
    label = f"**{row['symbol']}**\n💲{row['price']} | {row['change_pct']}%"
    if col.button(label, key=f"top{i}"):
        st.session_state["selected_stock"] = row["symbol"]
        st.session_state["search_input"] = ""

selected = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]

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
    return "✅ Buy - Expected to Increase" if forecast[-1] > df["Close"].iloc[-1] else "❌ Sell - Expected to Decrease"

# Chart
def plot_chart(symbol):
    df = get_stock_data(symbol)
    if df is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name=f"{symbol} Close", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="20-Day MA", line=dict(color="blue", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(500).mean(), name="500-Day MA", line=dict(color="red", dash="dot")))
    forecast = predict_next_30_days(df)
    if forecast.size > 0:
        future = pd.date_range(start=df.index[-1], periods=30, freq="D")
        fig.add_trace(go.Scatter(x=future, y=forecast, name="30-Day Forecast", line=dict(color="orange", dash="dash")))
    fig.update_layout(
        title=f"{symbol} Stock Forecast",
        paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font=dict(color="white"), legend=dict(bgcolor="#1E293B")
    )
    st.plotly_chart(fig, use_container_width=True)

plot_chart(selected)

# Info Boxes
df = get_stock_data(selected)
forecast = predict_next_30_days(df)
high = np.max(forecast) if forecast.size > 0 else None
curr = df["Close"].iloc[-1] if df is not None and not df.empty else None

if curr:
    st.markdown(f"<div class='info-box'>💲 Live Price: {curr:.4f}</div>", unsafe_allow_html=True)
if high:
    st.markdown(f"<div class='info-box'>📈 Highest Predicted: {high:.4f}</div>", unsafe_allow_html=True)
    diff = high - curr
    symbol = "▲" if diff > 0 else "▼"
    st.markdown(f"<div class='info-box'>📉 Change Forecast: {symbol} {diff:.2f}</div>", unsafe_allow_html=True)

st.markdown(f"<div class='info-box'>📊 Recommendation: {get_recommendation(df)}</div>", unsafe_allow_html=True)
