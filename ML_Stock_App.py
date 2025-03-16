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

def get_stock_data(stock_symbol):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    return hist

def get_index_data():
    """Fetches historical data for major stock indices (S&P 500, NASDAQ, Dow Jones)."""
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
    index_data = {name: yf.Ticker(symbol).history(period="1y") for name, symbol in indices.items()}
    return index_data

def predict_next_30_days(df):
    """Performs a simple linear regression to predict the next 30 days of stock prices."""
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    return pd.DataFrame({'Date': pd.date_range(start=df.index[-1], periods=30, freq='D'), 'Prediction': future_predictions})

def plot_stock_data(df, stock_symbol, future_df, index_data):
    """Generates a visualization of stock prices and index trends with clear formatting."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    # Stock price & forecast
    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    ax1.plot(future_df["Date"], future_df["Prediction"], label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=2)

    # Index trends on secondary y-axis
    for name, data in index_data.items():
        ax2.plot(data.index, data["Close"], linestyle="dotted", label=name, linewidth=1.5)

    # Labels & formatting
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Stock Price (USD)", color='white')
    ax2.set_ylabel("Index Values", color='white')

    # Tick formatting for readability
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='y', colors='white')

    # Legend Formatting
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

def get_most_traded_stock():
    """Finds the most traded stock based on daily volume."""
    sp500 = yf.Ticker("^GSPC").history(period="1d")
    if not sp500.empty:
        return "S&P 500", sp500.iloc[-1]['Close'], sp500.iloc[-1]['Volume']
    return "Unknown", None, None

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")

    # Display most traded stock of the day
    top_stock, top_price, top_volume = get_most_traded_stock()
    st.subheader(f"Most Traded Stock Today: {top_stock}")
    if top_price and top_volume:
        st.write(f"**Share Price:** {format_currency(top_price)}")
        st.write(f"**Volume:** {format_currency(top_volume)}")

    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL):")
    if st.button("Predict"):
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            index_data = get_index_data()
            future_df = predict_next_30_days(stock_data)
            plot_stock_data(stock_data, stock_symbol, future_df, index_data)

            # Company Performance Section
            stock_info = yf.Ticker(stock_symbol).info
            st.subheader(f"{stock_symbol.upper()} Company Performance")
            st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
            st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
            st.write(f"- **Net Income:** {format_currency(stock_info.get('netIncome', 0))}")
            st.write(f"- **Earnings Per Share (EPS):** {stock_info.get('trailingEps', 'N/A')}")
            st.write(f"- **Price-to-Earnings (P/E) Ratio:** {stock_info.get('trailingPE', 'N/A')}")

            # Prediction Recommendation
            st.subheader(f"Future Outlook for {stock_symbol.upper()}")
            if future_df["Prediction"].iloc[-1] > stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Buy** - The stock is predicted to rise in value over the next 30 days.")
            elif future_df["Prediction"].iloc[-1] < stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Sell** - The stock is predicted to decline, consider reducing holdings.")
            else:
                st.write("**Recommendation: Hold** - The stock is expected to remain stable.")
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()