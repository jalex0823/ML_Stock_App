import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import time
import requests
from textblob import TextBlob  # Sentiment Analysis

# 🌟 APPLY THEME STYLES
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    
    /* BUTTON STYLING */
    .stButton>button { 
        background-color: #F63366; 
        color: white; 
        border-radius: 8px; 
        padding: 10px 20px; 
        font-weight: bold; 
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover { background-color: #FF8AAE; transform: scale(1.05); }

    /* SEARCH BOX */
    .stTextInput>div>div>input { 
        border: 2px solid #F63366; 
        background-color: #1E293B; 
        color: white; 
        border-radius: 5px; 
    }
    .stTextInput>div>div>input:focus { border-color: #FF8AAE; }

    /* RADIO BUTTON */
    .stRadio>div>label>div>input { accent-color: #F63366; }

    /* CHECKBOX */
    .stCheckbox>label>div>input { accent-color: #F63366; }

    /* DARK MODE SUPPORT */
    .stMarkdown, .stText, .stTable, .stDataFrame { color: white !important; }

    /* ANIMATION */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeIn 0.8s ease-in-out; }
    </style>
    """, unsafe_allow_html=True)

# ✅ INITIALIZE SESSION STATE
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"  # Default top stock

# 📌 FETCH TOP STOCKS
def get_top_stocks():
    """Fetch the top 5 performing stocks dynamically."""
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

# 🎯 **SEARCH & SELECT STOCK**
st.markdown("<h3 style='color:#F63366;'>🔍 Search for a Stock</h3>", unsafe_allow_html=True)
search_stock = st.text_input("", key="search_input", placeholder="Enter stock symbol or company name...")

# 🎯 **SHOW TOP STOCKS AS BUTTONS**
st.markdown("<h3 style='color:#F63366;'>📊 Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(len(top_stocks))  # Align in a row

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]

# ✅ USE SEARCH OR SELECTED STOCK
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
    hist = ticker.history(period="1y")

    fig = go.Figure()

    # 🎯 STOCK PRICE LINE
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name=f"{stock_symbol} Close Price",
        line=dict(width=2, color="#F63366")
    ))

    # 📌 MOVING AVERAGES
    ma_50 = hist["Close"].rolling(window=50).mean()
    ma_200 = hist["Close"].rolling(window=200).mean()
    show_50_ma = st.checkbox(f"Show 50-Day MA for {stock_symbol}", value=True)
    show_200_ma = st.checkbox(f"Show 200-Day MA for {stock_symbol}", value=False)

    if show_50_ma:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=ma_50,
            mode="lines",
            name=f"{stock_symbol} 50-Day MA",
            line=dict(dash="dash", color="#FF8AAE")
        ))

    if show_200_ma:
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=ma_200,
            mode="lines",
            name=f"{stock_symbol} 200-Day MA",
            line=dict(dash="dash", color="red")
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

# 🎯 **DISPLAY STOCK CHART**
plot_stock_chart(selected_stock)

# ✅ **REAL-TIME STOCK PRICE UPDATES**
st.markdown("<h3 style='color:#F63366;'>📊 Real-Time Price Updates</h3>", unsafe_allow_html=True)
ticker = yf.Ticker(selected_stock)
price_placeholder = st.empty()

while True:
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
    price_placeholder.markdown(f"<h2 style='color:#F63366;'>💲 {current_price:.2f}</h2>", unsafe_allow_html=True)
    time.sleep(30)  # Update every 30 seconds

# ✅ **SHOW STOCK NEWS**
st.markdown("<h3 style='color:#F63366;'>📰 Latest News & Sentiment</h3>", unsafe_allow_html=True)
news = get_stock_news(selected_stock)
for item in news:
    st.markdown(f"<div class='news-card'><b>{item['title']}</b><br><span class='{item['sentiment'].lower()}'>{item['sentiment']}</span></div>", unsafe_allow_html=True)
