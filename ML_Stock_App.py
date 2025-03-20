import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests
from textblob import TextBlob  # Sentiment Analysis

# 🌟 APPLY THEME
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .btn { padding: 8px 15px; background: #F63366; color: white; border-radius: 5px; font-size: 14px; transition: 0.3s; }
    .btn:hover { background: #FC8181; transform: scale(1.1); }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    .news-card { padding: 10px; margin: 5px; background: #1E293B; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ✅ Initialize session state variables if they don't exist
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# 🔍 **SEARCH STOCK**
st.markdown("<h3 style='color:#F63366;'>🔍 Search a Stock</h3>", unsafe_allow_html=True)
search_stock = st.text_input("", key="search_input", placeholder="Type company name or symbol (e.g., Apple, AAPL)...")

# 📌 FETCH TOP 5 STOCKS
def get_top_stocks():
    """Fetch top performing stocks dynamically."""
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_data = []
    for stock in top_stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1d")
        price = hist["Close"].iloc[-1] if not hist.empty else "N/A"
        change_pct = ticker.info.get("52WeekChange", 0)
        change_amt = price * change_pct if price != "N/A" else "N/A"

        stock_data.append({
            "symbol": stock,
            "name": ticker.info.get("shortName", stock),
            "price": f"${price:.2f}" if price != "N/A" else "N/A",
            "change": f"{change_amt:.2f} ({change_pct:.2%})" if price != "N/A" else "N/A",
            "change_class": "positive" if change_pct > 0 else "negative"
        })
    return stock_data

# ✅ **TOP STOCK BUTTONS**
st.markdown("<h3 style='color:#F63366;'>Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(5)

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]
            st.session_state["search_input"] = ""

# ✅ **USE SEARCH OR TOP STOCK SELECTION**
selected_stock = search_stock if search_stock else st.session_state["selected_stock"]

# 📌 STOCK CHART
def plot_stock_chart(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    hist = ticker.history(period="1y")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"], mode="lines",
        name=f"{stock_symbol} Close Price", line=dict(width=2)
    ))

    fig.update_layout(
        title=f"{stock_symbol} Stock Price & Trends",
        xaxis_title="Date", yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font=dict(color="white"), legend=dict(bgcolor="#1E293B",
        bordercolor="white", borderwidth=1))

    st.plotly_chart(fig, use_container_width=True)

# 🎯 DISPLAY STOCK CHART
plot_stock_chart(selected_stock)

# ✅ REAL-TIME STOCK PRICE UPDATES
st.markdown("<h3 style='color:#F63366;'>📊 Real-Time Price Updates</h3>", unsafe_allow_html=True)
ticker = yf.Ticker(selected_stock)
price_placeholder = st.empty()

hist_data = ticker.history(period="1d")
if not hist_data.empty:
    current_price = hist_data["Close"].iloc[-1]
    price_placeholder.markdown(f"<h2 style='color:#F63366;'>💲 {current_price:.2f}</h2>", unsafe_allow_html=True)
else:
    price_placeholder.markdown(f"<h2 style='color:#F63366;'>⚠️ No price data available</h2>", unsafe_allow_html=True)

# ✅ STOCK NEWS & SENTIMENT
st.markdown("<h3 style='color:#F63366;'>📰 Stock News & Sentiment</h3>", unsafe_allow_html=True)
def get_stock_news(stock_symbol):
    """Fetch stock news and analyze sentiment."""
    api_url = f"https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(api_url)
    news_data = response.json().get("articles", [])[:5]
    for article in news_data:
        title = article["title"]
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        st.markdown(f"<div class='news-card'><b>{title}</b> - <span class='{sentiment_label.lower()}'>{sentiment_label}</span></div>", unsafe_allow_html=True)

get_stock_news(selected_stock)
