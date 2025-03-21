import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import requests
from textblob import TextBlob
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# ✅ **Initialize session state variables safely**
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

if "real_time_price" not in st.session_state:
    st.session_state["real_time_price"] = {}

# ✅ **Apply UI Theme**
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-btn { width: 180px; height: 50px; font-size: 14px; text-align: center; background: #1E40AF; 
                 color: white; border-radius: 5px; transition: 0.3s; cursor: pointer; border: none; }
    .stock-btn:hover { background: #3B82F6; transform: scale(1.05); }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    .news-card { padding: 10px; margin: 5px; background: #1E293B; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 📌 **Fetch S&P 500 Companies for Name-to-Symbol Search**
@st.cache_data
def get_sp500_list():
    """Loads S&P 500 companies for fuzzy matching."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        return table[['Security', 'Symbol']].dropna()
    except:
        return pd.DataFrame(columns=['Security', 'Symbol'])

sp500_list = get_sp500_list()

# 📌 **Get Stock Symbol from Company Name**
def get_stock_symbol(search_input):
    """Finds a stock symbol from either symbol input or company name."""
    search_input = search_input.strip().upper()
    
    # ✅ **Direct symbol match**
    if search_input in sp500_list['Symbol'].values:
        return search_input

    # ✅ **Try fuzzy matching with company name**
    result = process.extractOne(search_input, sp500_list['Security'])
    if result and result[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == result[0], 'Symbol'].values[0]
    
    return None  # No match found

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
st.markdown("<h3 style='color:white;'>🔍 Search by Company Name or Symbol</h3>", unsafe_allow_html=True)
search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or company name...").strip().upper()

# 📌 **Top Performing Stocks (Uniform Buttons)**
st.markdown("<h3 style='color:white;'>📈 Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
cols = st.columns(len(top_stocks))  # Evenly distribute buttons

for i, stock in enumerate(top_stocks):
    with cols[i]:
        if st.button(f"{stock['name']} ({stock['symbol']})", key=f"btn_{i}", help=f"Select {stock['name']}"):
            st.session_state["selected_stock"] = stock["symbol"]
            st.session_state["search_input"] = ""  # ✅ Auto-clear search field

# ✅ **Process Search Input**
selected_stock = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]

if not selected_stock:
    st.error("⚠️ Invalid company name or symbol. Please try again.")
    st.stop()

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
    """Simple Linear Regression Forecast for 30 Days"""
    if df is None or df.empty or len(df) < 30:
        return np.array([])
    
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]
    
    model = LinearRegression().fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    
    return model.predict(future_days)

# 📌 **Generate Buy/Sell Recommendation**
def get_recommendation(df):
    if df is None or df.empty:
        return "No Data Available"
    
    last_price = df["Close"].iloc[-1]
    forecast = predict_next_30_days(df)
    
    if forecast.size > 0 and forecast[-1] > last_price:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decline"

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

# 📌 **Display Recommendation**
st.markdown(f"<h3 style='color:white;'>📊 Recommendation: {get_recommendation(get_stock_data(selected_stock))}</h3>", unsafe_allow_html=True)

# ✅ **Real-Time Stock Updates**
st.markdown(f"<h3 style='color:white;'>💲 Live Price: {get_stock_data(selected_stock)['Close'].iloc[-1]:.2f}</h3>", unsafe_allow_html=True)
