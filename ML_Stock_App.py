import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import time
import requests
from textblob import TextBlob
from fuzzywuzzy import process  

# ✅ **Initialize session state variables safely**
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"
if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# ✅ **Apply UI Theme & Styles**
st.markdown("""
    <style>
    .search-container { display: flex; align-items: center; gap: 10px; }
    .search-input { flex-grow: 2; padding: 8px; border-radius: 5px; background: #1E293B; color: white; }
    .button-clear, .button-run { padding: 10px; border-radius: 5px; font-size: 14px; width: 100px; height: 38px; }
    .button-clear { background: #DC2626; color: white; }
    .button-clear:hover { background: #EF4444; }
    .button-run { background: #3B82F6; color: white; }
    .button-run:hover { background: #2563EB; }
    .stock-card { padding: 12px; margin: 5px; background: linear-gradient(135deg, #1E293B, #334155); 
                  border-radius: 10px; color: white; text-align: center; cursor: pointer; width: 100%; }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 📌 **Fetch S&P 500 Companies for Name-to-Symbol Search**
@st.cache_data
def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        return table[['Security', 'Symbol']].dropna()
    except:
        return pd.DataFrame(columns=['Security', 'Symbol'])

sp500_list = get_sp500_list()

# 📌 **Find Stock Symbol from Input**
def get_stock_symbol(search_input):
    search_input = search_input.strip().upper()
    if search_input in sp500_list['Symbol'].values:
        return search_input
    result = process.extractOne(search_input, sp500_list['Security'])
    if result and result[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == result[0], 'Symbol'].values[0]
    return None

# ✅ **Search Bar + Buttons (Clear & Run)**
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    search_input = st.text_input("", st.session_state["search_input"], placeholder="Type stock symbol or company name...", key="search_input")
with col2:
    if st.button("❌ Clear", key="clear_button"):
        st.session_state["search_input"] = ""
        st.rerun()
with col3:
    if st.button("▶️ Run", key="run_button"):
        st.session_state["selected_stock"] = get_stock_symbol(st.session_state["search_input"]) or st.session_state["search_input"]
        st.rerun()

# 📌 **Fetch Top 5 Performing Stocks**
def get_top_stocks():
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_data = []
    for stock in top_stocks:
        ticker = yf.Ticker(stock)
        stock_info = ticker.info
        price = stock_info.get("regularMarketPrice", 0)
        change_pct = stock_info.get("52WeekChange", 0)
        change_amt = price * change_pct
        stock_data.append({
            "symbol": stock,
            "name": stock_info.get("shortName", stock),
            "price": f"${price:.2f}",
            "change": f"{change_amt:.2f} ({change_pct:.2%})",
            "change_class": "positive" if change_pct > 0 else "negative"
        })
    return stock_data

# ✅ **Top Performing Stocks (Clickable)**
st.markdown("<h3 style='color:white;'>📈 Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(5)
for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]
            st.session_state["search_input"] = ""
            st.rerun()

# 📌 **Stock Data & Prediction**
def get_stock_data(stock_symbol):
    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="1y")
        if hist.empty:
            st.error(f"⚠️ No stock data found for '{stock_symbol}'. Please check the symbol.")
            return None
        return hist
    except Exception as e:
        st.error(f"⚠️ Error fetching data for '{stock_symbol}': {str(e)}")
        return None

def predict_next_30_days(df):
    if df is None or df.empty or len(df) < 30:
        return np.array([])
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    return model.predict(future_days)

# 📌 **Plot Stock Chart**
def plot_stock_chart(stock_symbol):
    hist = get_stock_data(stock_symbol)
    if hist is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name=f"{stock_symbol} Close Price", line=dict(width=2)))
    forecast = predict_next_30_days(hist)
    if forecast.size > 0:
        future_dates = pd.date_range(start=hist.index[-1], periods=30, freq="D")
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines", name=f"{stock_symbol} 30-Day Forecast", line=dict(dash="dash", color="orange")))
    fig.update_layout(title=f"{stock_symbol} Stock Price & Trends", xaxis_title="Date", yaxis_title="Stock Price (USD)", paper_bgcolor="#0F172A", plot_bgcolor="#0F172A", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

# 📌 **Display Stock Chart**
plot_stock_chart(st.session_state["selected_stock"])

# 📌 **Buy/Sell Recommendation**
def get_recommendation(df):
    if df is None or df.empty:
        return "No Data"
    last_price = df["Close"].iloc[-1]
    forecast = predict_next_30_days(df)
    if forecast.size > 0 and forecast[-1] > last_price:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decrease"

st.markdown(f"<h3 style='color:white;'>📉 Lowest Predicted Price: {min(predict_next_30_days(get_stock_data(st.session_state['selected_stock']))):.2f}</h3>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color:white;'>📢 Recommendation: {get_recommendation(get_stock_data(st.session_state['selected_stock']))}</h3>", unsafe_allow_html=True)
