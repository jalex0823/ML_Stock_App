import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replacing btalib with pandas_ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process
from functools import lru_cache

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
    """Search for a stock symbol using fuzzy matching on S&P 500 data."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        if "Security" not in table.columns or "Symbol" not in table.columns:
            raise ValueError("Error: 'Security' or 'Symbol' column missing from retrieved data.")
        company_list = table[['Security', 'Symbol']].dropna()
        company_list['Security'] = company_list['Security'].str.lower()
        
        if company_list.empty:
            st.error("Company list is empty. Unable to search for stock symbol.")
            return None
        
        result = process.extractOne(company_name.lower(), company_list['Security'])
        if result and result[1] >= 70:
            return company_list.loc[company_list['Security'] == result[0], 'Symbol'].values[0]
        else:
            st.error(f"No suitable match found for '{company_name}'. Try entering a different name.")
            return None
    except Exception as e:
        st.error(f"Failed to fetch company list. Error: {str(e)}")
        return None

def get_stock_data(stock_symbol):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    if hist.empty:
        st.error(f"No historical data found for {stock_symbol}.")
    return hist

@lru_cache(maxsize=5)
def get_index_data():
    """Fetches historical data for major stock indices (S&P 500, NASDAQ, Dow Jones)."""
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
    return {name: yf.Ticker(symbol).history(period="1y") for name, symbol in indices.items()}

def add_technical_indicators(df):
    """Calculates key technical indicators."""
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
    if df.empty or len(df) < 10:
        st.error("Insufficient data for prediction.")
        return np.array([])
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
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
    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        ax1.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=2)
    for name, data in index_data.items():
        ax2.plot(data.index, data["Close"], linestyle="dotted", label=name, linewidth=1.5)
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Stock Price (USD)", color='white')
    ax2.set_ylabel("Index Values", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax1.grid(color='gray', linestyle='dotted')
    legend1 = ax1.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    legend2 = ax2.legend(loc='upper right', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend1.get_texts():
        text.set_color("white")
    for text in legend2.get_texts():
        text.set_color("white")
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    company_name = st.text_input("Enter Company Name:")
    if st.button("Predict"):
        if company_name:
            stock_symbol = get_stock_symbol(company_name)
            if stock_symbol:
                stock_data = get_stock_data(stock_symbol)
                if not stock_data.empty:
                    stock_data = add_technical_indicators(stock_data)
                    index_data = get_index_data()
                    future_predictions = predict_next_30_days(stock_data)
                    plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
                else:
                    st.error("No data available for the selected company.")
            else:
                st.error("Unable to find stock symbol for the given company.")
        else:
            st.error("Please enter a valid company name.")

if __name__ == "__main__":
    main()
