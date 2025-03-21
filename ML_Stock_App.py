import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# ✅ **Initialize session state variables**
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# ✅ **Apply UI Theme & Button Styling**
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-btn-container { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
    .stock-btn { width: 180px; height: 50px; font-size: 14px; text-align: center; background: #1E40AF; 
                 color: white; border-radius: 5px; transition: 0.3s; cursor: pointer; border: none; padding: 10px 20px; }
    .stock-btn:hover { background: #3B82F6; transform: scale(1.05); }
    .selected-btn { background: #F63366; color: white; font-weight: bold; }
    .info-box { font-size: 18px; padding: 10px; background: #1E293B; color: white; border-radius: 5px; margin-top: 10px; }
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
    return [{"symbol": stock, "name": yf.Ticker(stock).info.get("shortName", stock)} for stock in top_stocks]

# 📌 **Search & Select Stock**
st.markdown("<h3 style='color:white;'>🔍 Search by Company Name or Symbol</h3>", unsafe_allow_html=True)
search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or company name...").strip().upper()

# 📌 **Top Performing Stocks (Uniform Buttons)**
st.markdown("<h3 style='color:white;'>📈 Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
col1, col2, col3, col4, col5 = st.columns(5)  # Ensure all buttons are aligned

for i, stock in enumerate(top_stocks):
    with [col1, col2, col3, col4, col5][i]:  # Map to respective columns
        button_label = f"{stock['name']} ({stock['symbol']})"
        is_selected = stock["symbol"] == st.session_state["selected_stock"]
        
        # ✅ Button Click: Clear Search Bar & Select Stock
        if st.button(button_label, key=f"btn_{i}", help="Click to select this stock", use_container_width=True):
            st.session_state["search_input"] = ""  # ✅ Auto-clear search field
            st.session_state["selected_stock"] = stock["symbol"]

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
        return hist if not hist.empty else None
    except:
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

# ✅ **Real-Time Stock Updates with Lowest Forecast Price**
df = get_stock_data(selected_stock)
forecast = predict_next_30_days(df)
lowest_forecast = np.min(forecast) if forecast.size > 0 else None
current_price = df["Close"].iloc[-1] if df is not None and not df.empty else None

st.markdown(f"<div class='info-box'>💲 Live Price: {current_price:.2f}</div>", unsafe_allow_html=True)
if lowest_forecast:
    st.markdown(f"<div class='info-box'>🔻 Lowest Predicted Price: {lowest_forecast:.2f}</div>", unsafe_allow_html=True)

# 📌 **Display Recommendation Below Lowest Predicted Price**
st.markdown(f"<div class='info-box'>📊 Recommendation: {get_recommendation(get_stock_data(selected_stock))}</div>", unsafe_allow_html=True)
