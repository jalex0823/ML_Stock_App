import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# Initialize session state
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"

if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

# Apply UI Theme
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-row { display: flex; align-items: center; justify-content: space-between;
                 background: #1E293B; padding: 10px; border-radius: 5px; margin-bottom: 5px; }
    .stock-name { font-size: 14px; font-weight: bold; color: white; flex: 2; }
    .stock-symbol { font-size: 12px; color: #888; flex: 1; text-align: left; }
    .stock-price { font-size: 14px; font-weight: bold; flex: 1; text-align: right; }
    .stock-change { font-size: 14px; font-weight: bold; flex: 1; text-align: right; }
    .positive { color: #16A34A; }
    .negative { color: #DC2626; }
    </style>
    """, unsafe_allow_html=True)

# Fetch S&P 500 list
@st.cache_data
def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        return table[['Security', 'Symbol']].dropna()
    except:
        return pd.DataFrame(columns=['Security', 'Symbol'])

sp500_list = get_sp500_list()

# Convert company name or symbol to stock symbol
def get_stock_symbol(search_input):
    search_input = search_input.strip().upper()
    if search_input in sp500_list['Symbol'].values:
        return search_input
    result = process.extractOne(search_input, sp500_list['Security'])
    if result and result[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == result[0], 'Symbol'].values[0]
    return None

# Fetch top 15 performing stocks dynamically
def get_top_stocks():
    top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "BABA", "JPM", "BA", "DIS", "V", "XOM"]
    stock_data = []
    
    for ticker in top_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("regularMarketPrice", 0)
        change_pct = info.get("52WeekChange", 0)
        change_amt = price * change_pct

        stock_data.append({
            "symbol": ticker,
            "name": info.get("shortName", ticker),
            "price": f"${price:.2f}",
            "change": f"{change_amt:.2f} ({change_pct:.2%})",
            "change_class": "positive" if change_pct > 0 else "negative",
            "mini_chart": stock.history(period="1mo")["Close"] if "Close" in stock.history(period="1mo") else None
        })

    return stock_data

# Display Top 15 Stocks
st.markdown("<h3 style='color:white;'>📈 Top 15 Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()

for stock in top_stocks:
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(f"<div class='stock-name'>{stock['name']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='stock-symbol'>{stock['symbol']}</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"<div class='stock-price'>{stock['price']}</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<div class='stock-change {stock['change_class']}'>{stock['change']}</div>", unsafe_allow_html=True)

    # Mini Sparkline Chart
    if stock["mini_chart"] is not None:
        mini_fig = go.Figure()
        mini_fig.add_trace(go.Scatter(y=stock["mini_chart"], mode="lines", line=dict(color="green" if stock["change_class"] == "positive" else "red")))
        mini_fig.update_layout(xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False), height=50, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(mini_fig, use_container_width=True)

# Determine selected stock
selected_stock = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]

if not selected_stock:
    st.error("⚠️ Invalid company name or symbol. Please try again.")
    st.stop()

# Fetch stock data
def get_stock_data(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1y")
        return data if not data.empty else None
    except:
        return None

# Predict next 30 days
def predict_next_30_days(df):
    if df is None or df.empty or len(df) < 30:
        return np.array([])
    df["Days"] = np.arange(len(df))
    model = LinearRegression().fit(df[["Days"]], df["Close"])
    return model.predict(np.arange(len(df), len(df)+30).reshape(-1, 1))

# Buy/Sell recommendation
def get_recommendation(df):
    forecast = predict_next_30_days(df)
    if df is None or df.empty or forecast.size == 0:
        return "No data available"
    if forecast[-1] > df["Close"].iloc[-1]:
        return "✅ Buy - Expected to Increase"
    return "❌ Sell - Expected to Decrease"

# Plot chart
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

# Display chart
plot_stock_chart(selected_stock)

# Show live price and recommendation
df = get_stock_data(selected_stock)
recommendation = get_recommendation(df)
st.markdown(f"<div class='info-box'>📊 Recommendation: {recommendation}</div>", unsafe_allow_html=True)
