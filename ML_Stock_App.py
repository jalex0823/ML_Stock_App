import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# ✅ Initialize session state
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# ✅ Apply UI Theme & Styling
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; justify-content: center; }
    .stock-box { background: #1E293B; padding: 15px; border-radius: 10px; width: 100%; height: 140px; text-align: center; display: flex; flex-direction: column; justify-content: center; }
    .stock-name { font-size: 14px; font-weight: bold; color: white; }
    .stock-symbol { font-size: 12px; color: #888; }
    .stock-price { font-size: 14px; font-weight: bold; }
    .stock-change { font-size: 14px; font-weight: bold; }
    .positive { color: #16A34A; }
    .negative { color: #DC2626; }
    .info-box { font-size: 16px; padding: 10px; background: #1E293B; color: white; border-radius: 5px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ✅ Fetch Real-Time Top 15 Performing Stocks
def get_top_gainers():
    try:
        tickers = yf.Ticker("^GSPC").history(period="1d")  # S&P 500 tickers
        ticker_list = tickers.index.to_list()[:100]  # Limit to first 100

        stocks = []
        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("regularMarketPrice", 0)
            change_pct = info.get("regularMarketChangePercent", 0)
            change_amt = price * (change_pct / 100)

            stocks.append({
                "symbol": ticker,
                "name": info.get("shortName", ticker),
                "price": f"${price:.2f}",
                "change": f"{change_amt:.2f} ({change_pct:.2f}%)",
                "change_class": "positive" if change_pct > 0 else "negative"
            })

        # ✅ Sort by highest gainers
        stocks = sorted(stocks, key=lambda x: float(x["change"].split()[0]), reverse=True)[:15]
        return stocks

    except Exception as e:
        st.error(f"⚠️ Error fetching top gainers: {str(e)}")
        return []

# ✅ Display Top 15 Stocks in 3 Columns
st.markdown("<h3 style='color:white;'>📈 Real-Time Top 15 Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_gainers()
cols = st.columns(3)

for i, stock in enumerate(top_stocks):
    with cols[i % 3]:  
        st.markdown(f"""
            <div class='stock-box'>
                <div class='stock-name'>{stock['name']}</div>
                <div class='stock-symbol'>{stock['symbol']}</div>
                <div class='stock-price'>{stock['price']}</div>
                <div class='stock-change {stock['change_class']}'>{stock['change']}</div>
            </div>
        """, unsafe_allow_html=True)

# ✅ Search input
search_input = st.text_input("Search by Company Name or Symbol", value=st.session_state["search_input"]).strip().upper()
selected_stock = search_input if search_input else st.session_state["selected_stock"]

# ✅ Fetch stock data
def get_stock_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1y")
        return data if not data.empty else None
    except:
        return None

# ✅ Predict next 30 days
def predict_next_30_days(df):
    if df is None or df.empty or len(df) < 30:
        return np.array([])
    df["Days"] = np.arange(len(df))
    model = LinearRegression().fit(df[["Days"]], df["Close"])
    return model.predict(np.arange(len(df), len(df)+30).reshape(-1, 1))

# ✅ Buy/Sell recommendation
def get_recommendation(df):
    forecast = predict_next_30_days(df)
    if df is None or df.empty or forecast.size == 0:
        return "No data available"
    if forecast[-1] > df["Close"].iloc[-1]:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decrease"

# ✅ Plot stock chart
def plot_stock_chart(symbol):
    df = get_stock_data(symbol)
    if df is None:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close
