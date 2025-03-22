import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression
import time

st.set_page_config(page_title="Real-Time Stock Forecast App", layout="wide")

# ---------- Style Theme ----------
st.markdown("""
    <style>
    body { background-color: #0F172A; color: white; }
    .stock-btn {
        width: 100%; height: 50px; font-size: 14px; 
        background: #1E40AF; color: white;
        border-radius: 5px; border: none; 
        transition: 0.3s; cursor: pointer;
    }
    .stock-btn:hover { background: #3B82F6; transform: scale(1.05); }
    .info-box {
        font-size: 16px; padding: 10px; background: #1E293B; 
        color: white; border-radius: 5px; margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- State Initialization ----------
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"
if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# ---------- Load S&P 500 Symbols ----------
@st.cache_data
def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    return table[['Security', 'Symbol']].dropna()

sp500_list = get_sp500_list()

def get_stock_symbol(search_input):
    search_input = search_input.strip().upper()
    if search_input in sp500_list['Symbol'].values:
        return search_input
    match = process.extractOne(search_input, sp500_list['Security'])
    if match and match[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == match[0], 'Symbol'].values[0]
    return None

# ---------- Get Top Gainers from S&P ----------
def get_top_gainers():
    tickers = sp500_list['Symbol'].tolist()[:300]  # Limit for speed
    changes = []
    for symbol in tickers:
        try:
            data = yf.Ticker(symbol).history(period="2d")
            if len(data) >= 2:
                change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
                changes.append((symbol, change))
        except:
            continue
    top = sorted(changes, key=lambda x: x[1], reverse=True)[:15]
    return [{"symbol": sym, "change": chg, "name": yf.Ticker(sym).info.get("shortName", sym)} for sym, chg in top]

# ---------- Layout: Search & Top Stocks ----------
st.markdown("### 🔍 Search by Company Name or Symbol")
search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or name...").strip().upper()

st.markdown("### 📈 Top Performing Stocks")
top_stocks = get_top_gainers()
col_layout = st.columns(3)

for i, stock in enumerate(top_stocks):
    with col_layout[i % 3]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"top_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]
            st.session_state["search_input"] = ""

# ---------- Determine Final Selection ----------
selected_stock = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]
if not selected_stock:
    st.error("Invalid company name or symbol.")
    st.stop()

# ---------- Stock Utilities ----------
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
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

def get_recommendation(df):
    forecast = predict_next_30_days(df)
    if df is None or df.empty or forecast.size == 0:
        return "No data available"
    if forecast[-1] > df["Close"].iloc[-1]:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decrease"

# ---------- Real-Time Animated Chart ----------
def render_chart(symbol):
    st.markdown(f"### 📊 {symbol} Real-Time Forecast Chart")
    chart_ph = st.empty()

    for _ in range(30):  # Simulate real-time for 15 minutes (30 x 30s)
        df = get_stock_data(symbol)
        forecast = predict_next_30_days(df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="cyan")))

        if forecast.size > 0:
            future_dates = pd.date_range(start=df.index[-1], periods=30, freq="D")
            fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines", name="30-Day Forecast", line=dict(color="orange", dash="dash")))

        fig.update_layout(
            xaxis_title="Date", yaxis_title="Price (USD)",
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="white"),
            legend=dict(bgcolor="#1E293B", bordercolor="white")
        )
        chart_ph.plotly_chart(fig, use_container_width=True)
        time.sleep(30)

# ---------- Render Chart ----------
render_chart(selected_stock)

# ---------- Display Stats ----------
df = get_stock_data(selected_stock)
forecast = predict_next_30_days(df)
highest_forecast = np.max(forecast) if forecast.size > 0 else None
current_price = df["Close"].iloc[-1] if df is not None and not df.empty else None

if current_price is not None:
    st.markdown(f"<div class='info-box'>💲 Live Price: ${current_price:.4f}</div>", unsafe_allow_html=True)

if highest_forecast:
    difference = highest_forecast - current_price
    st.markdown(f"<div class='info-box'>🔺 Highest Forecast Price: ${highest_forecast:.4f} (+${difference:.4f})</div>", unsafe_allow_html=True)

st.markdown(f"<div class='info-box'>📊 Recommendation: {get_recommendation(df)}</div>", unsafe_allow_html=True)