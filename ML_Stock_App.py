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
    """Finds the stock symbol based on the company name."""
    try:
        results = yf.search(company_name)
        if results and 'quotes' in results and len(results['quotes']) > 0:
            return results['quotes'][0]['symbol']
    except Exception as e:
        st.error(f"Error retrieving stock symbol: {e}")
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
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]  # MACD Line
        df['MACD_signal'] = macd.iloc[:, 1]  # Signal Line
    
    df.dropna(inplace=True)
    return df

def predict_next_30_days(df):
    """Performs a simple linear regression to predict the next 30 days of stock prices."""
    if df.empty or 'Close' not in df.columns or df['Close'].isnull().sum() > 0:
        return None  # No valid data available
    
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
    st.subheader(f"{stock_symbol} Stock Price & Market Trends")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    
    if future_predictions is not None:
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
    
    company_name = st.text_input("Enter Company Name (e.g., Apple Inc.):")
    stock_symbol = get_stock_symbol(company_name) if company_name else None
    
    if stock_symbol and st.button("Predict"):
        stock_data = get_stock_data(stock_symbol)
        stock_data = add_technical_indicators(stock_data)
        index_data = get_index_data()
        future_predictions = predict_next_30_days(stock_data)
        plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
        
        st.subheader(f"{stock_symbol} Company Performance")
        stock_info = yf.Ticker(stock_symbol).info
        st.write(f"- **Share Price:** {format_currency(stock_info.get('previousClose', 0))}")
        st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
        st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
        
        st.subheader(f"Future Outlook for {stock_symbol}")
        if future_predictions is not None and future_predictions[-1] > stock_data["Close"].iloc[-1]:
            st.write("**Recommendation: Buy** - Expected price increase in 30 days.")
        elif future_predictions is not None:
            st.write("**Recommendation: Sell** - Expected price decline in 30 days.")
        else:
            st.write("**Recommendation: Hold** - No significant change predicted.")

if __name__ == "__main__":
    main()