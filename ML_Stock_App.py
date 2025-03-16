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

def get_most_traded_stock():
    """Fetches the most highly traded stock of the day."""
    url = "https://finance.yahoo.com/most-active"
    df = pd.read_html(url)[0]  # Read the first table
    most_traded = df.iloc[0]  # Select the top most active stock

    ticker = most_traded["Symbol"]
    company_name = most_traded["Name"]
    volume = format_currency(most_traded["Volume"])
    last_price = format_currency(most_traded["Last Price"])
    share_price = format_currency(most_traded["Last Price"])

    return ticker, company_name, volume, last_price, share_price

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

def add_technical_indicators(df):
    """Calculates key technical indicators and handles missing values gracefully."""
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]  # MACD Line
        df['MACD_signal'] = macd.iloc[:, 1]  # Signal Line

    bollinger = ta.bbands(df['Close'], length=20)
    if bollinger is not None:
        df['Bollinger_Upper'] = bollinger.iloc[:, 0]
        df['Bollinger_Middle'] = bollinger.iloc[:, 1]
        df['Bollinger_Lower'] = bollinger.iloc[:, 2]

    df.dropna(inplace=True)
    return df

def predict_next_30_days(df):
    """Performs a simple linear regression to predict the next 30 days of stock prices."""
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    return future_predictions

def plot_stock_data(df, stock_symbol, future_predictions, index_data):
    """Generates a visualization of stock prices and index trends with clear formatting."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    ax1.plot(df.index, df["SMA_50"], label="50-day SMA", linestyle="dashed", color="orange", linewidth=2)
    ax1.plot(df.index, df["SMA_200"], label="200-day SMA", linestyle="dashed", color="red", linewidth=2)

    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    ax1.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=2)

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
    try:
        top_ticker, top_name, top_volume, top_last_price, top_share_price = get_most_traded_stock()
        st.subheader(f"Most Traded Stock Today: {top_name} ({top_ticker})")
        st.write(f"- **Last Price:** {top_last_price}")
        st.write(f"- **Trading Volume:** {top_volume}")
        st.write(f"- **Share Price:** {top_share_price}")
    except:
        st.write("⚠️ Unable to fetch the most traded stock at the moment.")
    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL):")
    if st.button("Predict"):
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_data = add_technical_indicators(stock_data)
            index_data = get_index_data()
            future_predictions = predict_next_30_days(stock_data)
            plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()