import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

# 🌟 Streamlit UI Customization - MSN Watchlist Theme
st.set_page_config(page_title="Stock Recommender", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    body {background-color: #0F172A; color: white;}
    .stDataFrame {background-color: #1E293B !important; color: white;}
    .stButton>button {background-color: #0078D4 !important; color: white !important;}
    .stTextInput>div>div>input, .stSelectbox>div>div>div>input {background-color: #1E293B !important; color: white !important; border-radius: 5px; padding: 10px;}
    .stock-card {background-color: #1E293B; border-radius: 8px; padding: 10px; margin-bottom: 8px;}
    .stock-title {color: white; font-size: 16px; font-weight: bold;}
    .stock-metrics {color: #94A3B8; font-size: 14px;}
    .scroll-container {max-height: 300px; overflow-y: auto;}
    </style>
    """,
    unsafe_allow_html=True
)

# 🔥 Format currency display
def format_currency(value):
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"

# 🔍 Get top 5 performing stocks dynamically
def get_top_stocks():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        stock_price = stock_info.get("regularMarketPrice", 0)
        ytd_change = stock_info.get("52WeekChange", 0)
        ytd_change_amt = stock_price * ytd_change
        stock_data.append((stock_info.get("longName", ticker), ticker, stock_price, ytd_change_amt, ytd_change))
    
    return sorted(stock_data, key=lambda x: x[2], reverse=True)[:5]

# 🔄 Get stock symbol from company name
def get_stock_symbol(company_name):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        company_list = table[['Security', 'Symbol']].dropna()
        company_list['Security'] = company_list['Security'].str.lower()
        result = process.extractOne(company_name.lower(), company_list['Security'])
        return company_list.loc[company_list['Security'] == result[0], 'Symbol'].values[0] if result and result[1] >= 70 else None
    except Exception:
        return None

# 📊 Get stock historical data
def get_stock_data(stock_symbol, period="1y"):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period=period)
    return hist if not hist.empty else None

# 📈 Predict next 30 days
def predict_next_30_days(df):
    if df.empty or len(df) < 10:
        return np.array([])
    df['Days'] = np.arange(len(df))
    model = LinearRegression().fit(df[['Days']], df['Close'])
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

# 📊 Plot stock chart with forecast
def plot_stock_data(df, stock_symbol, future_predictions):
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    ax.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        ax.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="orange", linewidth=2)

    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Stock Price (USD)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax.grid(color='gray', linestyle='dotted')

    legend = ax.legend(loc='upper left', fontsize='small', facecolor='#1E293B', framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    st.pyplot(fig)

# 🌟 MAIN APP LAYOUT
def main():
    col1, col2 = st.columns([1, 3])  # Sidebar | Main Content

    # 📌 Sidebar (Scrollable Top 5 Stocks Watchlist)
    with col1:
        st.markdown("<h3 style='color:white;'>Top 5 Stocks</h3>", unsafe_allow_html=True)
        top_stocks = get_top_stocks()

        # Scrollable section
        st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
        for name, symbol, price, ytd_amt, ytd_pct in top_stocks:
            color = "green" if ytd_pct > 0 else "red"
            if st.button(f"{name} ({symbol})"):
                st.session_state["selected_stock"] = symbol

            st.markdown(
                f"<div class='stock-card'>"
                f"<strong class='stock-title'>{name} ({symbol})</strong><br>"
                f"<span class='stock-metrics'>Price: {format_currency(price)}</span><br>"
                f"<span style='color:{color}; font-weight:bold;'>YTD: {format_currency(ytd_amt)} ({ytd_pct:.2%})</span>"
                f"</div>", 
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # 📌 Main Section (Chart & Details)
    with col2:
        period = st.selectbox("Select Time Period", ["1d", "5d", "1mo", "6mo", "ytd", "1y", "5y", "max"])
        stock_input = st.text_input("Enter Company Name:", value=st.session_state.get("selected_stock", ""))
        
        if st.button("Search"):
            stock_symbol = get_stock_symbol(stock_input)
            if stock_symbol:
                stock_data = get_stock_data(stock_symbol, period)
                if stock_data is not None:
                    future_predictions = predict_next_30_days(stock_data)
                    plot_stock_data(stock_data, stock_symbol, future_predictions)

if __name__ == "__main__":
    main()
