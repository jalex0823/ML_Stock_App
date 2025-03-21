import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# ✅ Initialize session state variables
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# ✅ Apply UI Theme & Styling
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .btn-container { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
    .stock-btn { width: 180px; height: 50px; font-size: 14px; text-align: center; background: #1E40AF; 
                 color: white; border-radius: 5px; transition: 0.3s; cursor: pointer; border: none; padding: 10px 20px; }
    .stock-btn:hover { background: #3B82F6; transform: scale(1.05); }
    .selected-btn { background: #F63366; color: white; font-weight: bold; }
    .info-box { font-size: 18px; padding: 10px; background: #1E293B; color: white; border-radius: 5px; margin-top: 10px; }
    .clear-btn { background: #F63366; color: white; border-radius: 5px; padding: 8px 15px; border: none; cursor: pointer; }
    .run-btn { background: #3B82F6; color: white; border-radius: 5px; padding: 8px 15px; border: none; cursor: pointer; }
    </style>
    """, unsafe_allow_html=True)

# 📌 Fetch S&P 500 Companies for Name-to-Symbol Search
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

# 📌 Get Stock Symbol from Company Name
def get_stock_symbol(search_input):
    """Finds a stock symbol from either symbol input or company name."""
    search_input = search_input.strip().upper()

    if search_input in sp500_list['Symbol'].values:
        return search_input

    result = process.extractOne(search_input, sp500_list['Security'])
    if result and result[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == result[0], 'Symbol'].values[0]

    return None  

# 📌 Fetch Top 5 Performing Stocks
def get_top_stocks():
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    return [{"symbol": stock, "name": yf.Ticker(stock).info.get("shortName", stock)} for stock in top_stocks]

# ✅ Search & Select Stock
st.markdown("<h3 style='color:white;'>🔍 Search by Company Name or Symbol</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 1, 1])  
with col1:
    search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or company name...").strip().upper()
with col2:
    if st.button("❌ Clear", help="Clear the search box", key="clear-btn", use_container_width=True):
        st.session_state["search_input"] = ""
with col3:
    if st.button("▶ Run", help="Fetch stock data", key="run-btn", use_container_width=True):
        st.session_state["selected_stock"] = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]

# 📌 Top Performing Stocks (Uniform Buttons)
st.markdown("<h3 style='color:white;'>📈 Top Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()
col1, col2, col3, col4, col5 = st.columns(5)

for i, stock in enumerate(top_stocks):
    with [col1, col2, col3, col4, col5][i]:
        button_label = f"{stock['name']} ({stock['symbol']})"
        if st.button(button_label, key=f"btn_{i}", use_container_width=True):
            st.session_state["search_input"] = ""
            st.session_state["selected_stock"] = stock["symbol"]

# ✅ Process Search Input
selected_stock = st.session_state["selected_stock"]

# 📌 Stock Data & Prediction
def get_stock_data(stock_symbol):
    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="1y")
        return hist if not hist.empty else None
    except:
        return None

def predict_next_30_days(df):
    if df is None or df.empty or len(df) < 30:
        return np.array([])

    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]

    model = LinearRegression().fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)

    return model.predict(future_days)

# ✅ Generate Buy/Sell Recommendation
def get_recommendation(df):
    if df is None or df.empty:
        return "⚠️ No Data Available"

    last_price = df["Close"].iloc[-1]
    forecast = predict_next_30_days(df)

    if forecast.size > 0:
        last_forecast = forecast[-1]

        if last_forecast > last_price:
            return "✅ Buy - Expected to Increase"
        else:
            return "❌ Sell - Expected to Decline"

    return "⚠️ Not Enough Data for Prediction"

# 📌 Plot Stock Chart with 20-day & 500-day Moving Averages
def plot_stock_chart(stock_symbol):
    hist = get_stock_data(stock_symbol)
    
    if hist is None:
        return  

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["Close"],
        mode="lines",
        name=f"{stock_symbol} Close Price",
        line=dict(width=2)
    ))

    # Add 20-Day and 500-Day Moving Averages
    hist["MA_20"] = hist["Close"].rolling(window=20).mean()
    hist["MA_500"] = hist["Close"].rolling(window=500).mean()

    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA_20"], mode="lines", name="20-Day MA", line=dict(color="blue", dash="dot")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA_500"], mode="lines", name="500-Day MA", line=dict(color="red", dash="dot")))

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

# 📌 Display Stock Chart
plot_stock_chart(selected_stock)

# ✅ Show Real-Time Stock Updates & Recommendation
df = get_stock_data(selected_stock)
st.markdown(f"<div class='info-box'>💲 Live Price: {df['Close'].iloc[-1]:.2f}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='info-box'>📊 Recommendation: {get_recommendation(df)}</div>", unsafe_allow_html=True)
