import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
from fuzzywuzzy import fuzz

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st_autorefresh(interval=600 * 1000, key="realtime_top_refresh")

if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"
if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

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

st.markdown("<h1 style='color:white; text-align:center;'>🧠 The AI Predictive Stock Application</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color:white;'>🔍 Search by Symbol or Company Name</h3>", unsafe_allow_html=True)
col_search, col_clear = st.columns([6, 1])
with col_search:
    search_input = st.text_input(
        "Stock search input (hidden)", key="search_input_box",
        value=st.session_state["search_input"],
        placeholder="e.g. AAPL, Tesla, SHOP.TO, BMW, Alibaba",
        label_visibility="collapsed"
    ).strip()
with col_clear:
    if st.button("❌", help="Clear search", use_container_width=True):
        st.session_state["search_input"] = ""
        search_input = ""
@st.cache_data(ttl=600)
def resolve_symbol_from_input(search_input):
    query = search_input.strip().upper()
    try:
        info = yf.Ticker(query).info
        if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
            return query
    except:
        pass

    candidates = [
        "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "BABA", "NVDA", "NFLX", "DIS",
        "SHOP.TO", "TD.TO", "RY.TO", "BMW.DE", "SIE.DE", "SONY", "6758.T", "TCEHY", "VOD.L", "BP.L"
    ]

    best_match = None
    best_score = 0

    for symbol in candidates:
        try:
            info = yf.Ticker(symbol).info
            name = info.get("longName", "")
            score = fuzz.token_set_ratio(search_input.lower(), name.lower())
            if name and score > best_score and "regularMarketPrice" in info:
                best_score = score
                best_match = symbol
        except:
            continue

    return best_match


def get_index_summary():
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC"
    }
    summary = {}
    for name, symbol in indices.items():
        try:
            info = yf.Ticker(symbol).info
            price = info.get("regularMarketPrice", 0)
            change = info.get("regularMarketChange", 0)
            percent = info.get("regularMarketChangePercent", 0)
            arrow = "▲" if change >= 0 else "▼"
            summary[name] = f"{arrow} {price:.2f} ({change:+.2f}, {percent:+.2f}%)"
        except:
            summary[name] = "Data unavailable"
    return summary


@st.cache_data(ttl=60)
def get_top_stocks():
    tickers = [
        "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "UNH", "PEP",
        "V", "MA", "JPM", "WMT", "HD", "DIS", "PFE", "BAC", "INTC", "COST", "TMO",
        "CRM", "CVX", "KO", "ABT", "ABBV", "AMD", "MCD", "ADBE", "QCOM", "MRK", "T", "BA",
        "NKE", "XOM", "GS", "GE", "LMT", "ORCL", "SBUX", "MDT", "HON", "ISRG", "VRTX",
        "WBA", "LOW", "ZTS", "BLK", "BMY", "CAT"
    ]
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            price = info.get("regularMarketPrice", 0)
            change = info.get("regularMarketChange", 0)
            percent = info.get("regularMarketChangePercent", 0)
            data.append({
                "symbol": t,
                "name": info.get("shortName", t),
                "price": price,
                "change": change,
                "percent": percent
            })
        except:
            continue
    return sorted(data, key=lambda x: x["percent"], reverse=True)[:15]


# Display Index Summary
index_summary = get_index_summary()
st.markdown(f"""
    <div class='info-box' style="display: flex; gap: 40px; align-items: center;">
        <span>📊 <b>S&P 500:</b> {index_summary['S&P 500']}</span>
        <span>📊 <b>Dow Jones:</b> {index_summary['Dow Jones']}</span>
        <span>📊 <b>NASDAQ:</b> {index_summary['NASDAQ']}</span>
    </div>
""", unsafe_allow_html=True)


# Top Stocks Section
st.markdown("<h3 style='color:white;'>📈 Top 15 Performing Stocks (Real-Time)</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
col1, col2, col3 = st.columns(3)
for i, stock in enumerate(top_stocks):
    col = [col1, col2, col3][i % 3]
    with col:
        label = f"**{stock['name']}**\n{stock['symbol']}\n💲{stock['price']:.2f}\n📈 {stock['change']:+.2f} ({stock['percent']:.2f}%)"
        if st.button(label, key=f"top_{i}", use_container_width=True):
            st.session_state["search_input"] = ""
            st.session_state["selected_stock"] = stock["symbol"]
selected_stock = resolve_symbol_from_input(search_input) if search_input else st.session_state["selected_stock"]
if not selected_stock:
    st.error("⚠️ Could not find a matching stock symbol. Try again.")
    st.stop()
else:
    st.session_state["selected_stock"] = selected_stock


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
    future_days = pd.DataFrame({"Days": np.arange(len(df), len(df) + 30)})
    return model.predict(future_days)


def get_recommendation(df):
    forecast = predict_next_30_days(df)
    if df is None or df.empty or forecast.size == 0:
        return "No data available"
    return "✅ Buy - Expected to Increase" if forecast[-1] > df["Close"].iloc[-1] else "❌ Sell - Expected to Decrease"


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


# Display the chart and predictions
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
