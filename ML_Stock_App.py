import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replacing btalib with pandas_ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process  # Install with pip install fuzzywuzzy

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
    """
    Searches for the best matching stock symbol based on a given company name.
    Uses fuzzy matching to compare with a list of known stock symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    company_list = table[['Security', 'Symbol']].dropna()

    best_match, score = process.extractOne(company_name, company_list['Security'])
    
    if score > 75:  # Ensure match is reasonably close
        stock_symbol = company_list.loc[company_list['Security'] == best_match, 'Symbol'].values[0]
        return stock_symbol
    else:
        return None

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
    company_name = st.text_input("Enter Company Name:")
    if st.button("Search & Predict"):
        stock_symbol = get_stock_symbol(company_name)
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_data = add_technical_indicators(stock_data)
            index_data = get_index_data()
            future_predictions = predict_next_30_days(stock_data)
            plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
        else:
            st.error("Company name not found. Please try again.")

if __name__ == "__main__":
    main()