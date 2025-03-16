import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replacing btalib with pandas_ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression

def format_currency(value):
    """Formats numbers into readable currency denominations."""
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"

def get_stock_symbol(company_name):
    """Fetches the stock symbol for a given company name."""
    try:
        search_result = yf.Ticker(company_name)
        return search_result.ticker
    except:
        return None

def get_stock_data(stock_symbol):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    return hist

def get_top_traded_stock():
    """Fetches the most actively traded stock of the day."""
    try:
        trending = yf.Ticker("SPY").history(period="1d")
        return "SPY", trending["Close"].iloc[-1]
    except:
        return None, None

def get_index_data():
    """Fetches historical data for major stock indices (S&P 500, NASDAQ, Dow Jones)."""
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
    index_data = {name: yf.Ticker(symbol).history(period="1y") for name, symbol in indices.items()}
    return index_data

def predict_next_30_days(df):
    """Performs a simple linear regression to predict the next 30 days of stock prices."""
    if df.empty or "Close" not in df.columns:
        return []
    df["Days"] = np.arange(len(df))
    X = df[["Days"]]
    y = df["Close"]
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    return model.predict(future_days)

def plot_stock_data(df, stock_symbol, future_predictions, index_data):
    """Generates a visualization of stock prices and index trends with clear formatting."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    
    # Stock price & prediction
    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    if future_predictions:
        future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
        ax1.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=2)
    
    # Index trends on secondary y-axis
    for name, data in index_data.items():
        ax2.plot(data.index, data["Close"], linestyle="dotted", label=name, linewidth=1.5)
    
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Stock Price (USD)", color='white')
    ax2.set_ylabel("Index Values", color='white')
    
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    legend1 = ax1.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    legend2 = ax2.legend(loc='upper right', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    
    for text in legend1.get_texts():
        text.set_color("white")
    for text in legend2.get_texts():
        text.set_color("white")
    
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    
    ax1.grid(color='gray', linestyle='dotted')
    
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    
    # Display top-traded stock
    top_stock, top_price = get_top_traded_stock()
    if top_stock and top_price:
        st.subheader(f"Most Actively Traded Stock: {top_stock}")
        st.write(f"**Share Price:** {format_currency(top_price)}")
    
    company_name = st.text_input("Enter Company Name (e.g., Apple Inc.):")
    if st.button("Predict"):
        if company_name:
            stock_symbol = get_stock_symbol(company_name)
            if not stock_symbol:
                st.error("Company name not found. Please try again.")
                return
            stock_data = get_stock_data(stock_symbol)
            index_data = get_index_data()
            future_predictions = predict_next_30_days(stock_data)
            plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
            
            # Display company details
            st.subheader(f"{stock_symbol.upper()} Company Performance")
            stock_info = yf.Ticker(stock_symbol).info
            st.write(f"- **Share Price:** {format_currency(stock_data['Close'].iloc[-1])}")
            st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
            st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
            st.write(f"- **Net Income:** {format_currency(stock_info.get('netIncome', 0))}")
            st.write(f"- **Earnings Per Share (EPS):** {stock_info.get('trailingEps', 'N/A')}")
            st.write(f"- **Price-to-Earnings (P/E) Ratio:** {stock_info.get('trailingPE', 'N/A')}")
            
            # Prediction recommendation
            st.subheader(f"Future Outlook for {stock_symbol.upper()}")
            if future_predictions and future_predictions[-1] > stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Buy** - The stock is predicted to rise in value over the next 30 days.")
            elif future_predictions and future_predictions[-1] < stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Sell** - The stock is predicted to decline, consider reducing holdings.")
            else:
                st.write("**Recommendation: Hold** - The stock is expected to remain stable.")
        else:
            st.error("Please enter a valid company name.")

if __name__ == "__main__":
    main()