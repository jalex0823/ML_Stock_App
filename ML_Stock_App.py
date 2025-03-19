import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import time
import requests
from textblob import TextBlob  # Sentiment Analysis

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
    .watchlist-table th, .watchlist-table td { padding: 10px; text-align: left; border: 1px solid #334155; }
    .watchlist-table th { background-color: #1E293B; color: white; font-size: 14px; }
    .watchlist-table td { background-color: #0F172A; color: white; font-size: 13px; }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    .news-card { padding: 10px; margin: 5px; background: #1E293B; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 📌 FETCH TOP STOCKS
def get_top_stocks():
    """Fetch top 5 performing stocks dynamically."""
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_data = []
    for stock in top_stocks:
        ticker = yf.Ticker(stock)
        price = ticker.info.get("regularMarketPrice", 0)
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

# ✅ AUTO-LOAD TOP STOCK
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

# 📌 MOVING SEARCH TO THE TOP
st.markdown("<h3 style='color:white;'>🔍 Search a Stock</h3>", unsafe_allow_html=True)
search_stock = st.text_input("", key="search_input", placeholder="Type stock symbol (e.g., TSLA, MSFT)...")

# ✅ ALLOW TOP STOCK CLICKABILITY
st.markdown("<h3 style='color:white;'>Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(5)  # Align in a row

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]

# ✅ USE SEARCH OR TOP STOCK SELECTION
selected_stock = search_stock if search_stock else st.session_state["selected_stock"]

# 📌 STOCK NEWS & SENTIMENT ANALYSIS
def get_stock_news(stock_symbol):
    """Fetch news headlines and perform sentiment analysis."""
    api_url = f"https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(api_url)
    news_data = response.json().get("articles", [])[:5]  # Get top 5 news articles

    news_results = []
    for article in news_data:
        title = article["title"]
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        news_results.append({
            "title": title,
            "sentiment": sentiment_label
        })
    
    return news_results

# 📌 LOAD STOCK DATA & DISPLAY GRAPH
def plot_stock_chart(stock_symbol):
    ticker = yf.Ticker(stock_symbol)
    hist = ticker.history(period="6mo")

    fig = go.Figure()

    # 🎯 Stock Price Line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name=f"{stock_symbol} Close Price",
        line=dict(width=2)
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

# ✅ REAL-TIME STOCK PRICE UPDATES
st.markdown("<h3 style='color:white;'>📊 Real-Time Price Updates</h3>", unsafe_allow_html=True)
ticker = yf.Ticker(selected_stock)
price_placeholder = st.empty()

while True:
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    price_placeholder.markdown(f"<h2 style='color:white;'>💲 {current_price:.2f}</h2>", unsafe_allow_html=True)
    time.sleep(30)  # Update every 30 seconds

# ✅ SHOW STOCK NEWS
st.markdown("<h3 style='color:white;'>📰 Latest News & Sentiment</h3>", unsafe_allow_html=True)
news = get_stock_news(selected_stock)
for item in news:
    st.markdown(f"<div class='news-card'><b>{item['title']}</b><br><span class='{item['sentiment'].lower()}'>{item['sentiment']}</span></div>", unsafe_allow_html=True)
