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
    .stock-box { background: #1E293B; padding: 15px; border-radius: 10px; text-align: center; width: 100%; height: 140px; display: flex; flex-direction: column; justify-content: center; }
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
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "AMD", "BA", "V", "DIS", "KO", "WMT"]
        stocks = []

        for ticker in tickers[:15]:  # Limit to 15 stocks
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
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{symbol} Close Price", line=dict(width=2)))

    # Add moving averages
    ma_20 = df["Close"].rolling(window=20).mean()
    ma_500 = df["Close"].rolling(window=500).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma_20, mode="lines", name="20-Day MA", line=dict(color="blue", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=ma_500, mode="lines", name="500-Day MA", line=dict(color="red", dash="dot")))

    # Add forecast
    forecast = predict_next_30_days(df)
    if forecast.size > 0:
        future_dates = pd.date_range(start=df.index[-1], periods=30, freq="D")
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode="lines", name="30-Day Forecast", line=dict(dash="dash", color="orange")))

    fig.update_layout(
        title=f"{symbol} Stock Price & Trends",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="white"),
        legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ✅ Display stock chart
plot_stock_chart(selected_stock)

# ✅ Show live price and recommendation
df = get_stock_data(selected_stock)
forecast = predict_next_30_days(df)
highest_forecast = np.max(forecast) if forecast.size > 0 else None  # CHANGED: Show highest instead of lowest
current_price = df["Close"].iloc[-1] if df is not None and not df.empty else None

if current_price is not None:
    st.markdown(f"<div class='info-box'>💲 Live Price: {current_price:.4f}</div>", unsafe_allow_html=True)

if highest_forecast:
    st.markdown(f"<div class='info-box'>🔺 Highest Predicted Price (Next 30 Days): {highest_forecast:.4f}</div>", unsafe_allow_html=True)  # CHANGED

recommendation = get_recommendation(df)
st.markdown(f"<div class='info-box'>📊 Recommendation: {recommendation}</div>", unsafe_allow_html=True)
