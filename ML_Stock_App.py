import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import time
import requests
from textblob import TextBlob  # Sentiment Analysis

# ✅ **Initialize session state variables safely**
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"  # Default to Apple

# ✅ **Apply UI Theme**
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-card { padding: 15px; margin: 5px; background: linear-gradient(135deg, #1E293B, #334155); 
                  border-radius: 10px; color: white; text-align: center; transition: 0.3s; cursor: pointer; }
    .stock-card:hover { transform: scale(1.05); background: linear-gradient(135deg, #334155, #475569); }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

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

# 📌 **Search & Select Stock**
st.markdown("<h3 style='color:white;'>🔍 Search a Stock</h3>", unsafe_allow_html=True)
search_stock = st.text_input("", placeholder="Type stock symbol (e.g., TSLA, MSFT)...").strip().upper()

# 📌 **Top Performing Stocks (Clickable)**
st.markdown("<h3 style='color:white;'>Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(5)  # Align in a row

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}"):
            st.session_state["selected_stock"] = stock["symbol"]

# ✅ **Use Either Search or Top Stock Selection**
selected_stock = search_stock if search_stock else st.session_state["selected_stock"]

# 📌 **Stock Data & Prediction**
def get_stock_data(stock_symbol):
    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="1y")
        
        # ✅ Ensure the stock symbol is valid before proceeding
        if hist.empty:
            st.error(f"⚠️ No stock data found for '{stock_symbol}'. Please check the symbol.")
            return None

        return hist
    except Exception as e:
        st.error(f"⚠️ Error fetching data for '{stock_symbol}': {str(e)}")
        return None

def predict_next_30_days(df):
    """Simple Linear Regression Forecast for 30 Days"""
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
        return  # Stop execution if data is missing

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name=f"{stock_symbol} Close Price",
        line=dict(width=2)
    ))

    # 🎯 **Add 30-Day Forecast**
    forecast = predict_next_30_days(hist)
    if forecast.size > 0:
        future_dates = pd.date_range(start=hist.index[-1], periods=30, freq="D")
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecast,
            mode="lines",
            name=f"{stock_symbol} 30-Day Forecast",
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

# 📌 **Display Stock Chart**
plot_stock_chart(selected_stock)

# ✅ **Buy/Sell Recommendation**
def get_recommendation(df):
    if df is None or df.empty:
        return "No Data"
    
    last_price = df["Close"].iloc[-1]
    forecast = predict_next_30_days(df)
    
    if forecast.size > 0 and forecast[-1] > last_price:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decrease"

st.markdown(f"<h3 style='color:white;'>Recommendation: {get_recommendation(get_stock_data(selected_stock))}</h3>", unsafe_allow_html=True)

# 📌 **Stock News & Sentiment Analysis**
def get_stock_news(stock_symbol):
    api_url = f"https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(api_url)
    news_data = response.json().get("articles", [])[:5]

    news_results = []
    for article in news_data:
        title = article["title"]
        sentiment = TextBlob(title).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        news_results.append(f"<div class='news-card'><b>{title}</b> - {sentiment_label}</div>")
    
    return news_results

# ✅ **Display News**
st.markdown("<h3 style='color:white;'>📰 Latest News</h3>", unsafe_allow_html=True)
for news in get_stock_news(selected_stock):
    st.markdown(news, unsafe_allow_html=True)
