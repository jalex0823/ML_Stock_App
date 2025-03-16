import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replacing btalib with pandas_ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

def format_currency(value):
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def get_index_data():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
    index_data = {}
    for name, symbol in indices.items():
        index = yf.Ticker(symbol)
        index_data[name] = index.history(period="1y")
    return index_data

def add_technical_indicators(df):
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    macd = ta.macd(df['Close'])
    if isinstance(macd, pd.DataFrame) and 'MACD_12_26_9' in macd.columns:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
    else:
        df['MACD'] = np.nan
        df['MACD_signal'] = np.nan
    
    bollinger = ta.bbands(df['Close'], length=20)
    if isinstance(bollinger, pd.DataFrame) and 'BBU_20_2.0' in bollinger.columns:
        df['Bollinger_Upper'] = bollinger['BBU_20_2.0']
        df['Bollinger_Middle'] = bollinger['BBM_20_2.0']
        df['Bollinger_Lower'] = bollinger['BBL_20_2.0']
    else:
        df['Bollinger_Upper'] = np.nan
        df['Bollinger_Middle'] = np.nan
        df['Bollinger_Lower'] = np.nan

    df.dropna(inplace=True)
    return df

def predict_next_30_days(df):
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    future_predictions = model.predict(future_days)
    return future_predictions

def plot_stock_data(df, ticker, future_predictions, index_data):
    st.subheader(f"Stock Price Visualization and Market Index Trends for {ticker}")
    
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
    
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_currency(x)))
    
    legend1 = ax1.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    legend2 = ax2.legend(loc='upper right', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend1.get_texts():
        text.set_color("white")
    for text in legend2.get_texts():
        text.set_color("white")
    
    ax1.grid(color='gray', linestyle='dotted')
    
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")
    if st.button("Predict"):
        if ticker:
            stock_data = get_stock_data(ticker)
            stock_data = add_technical_indicators(stock_data)
            index_data = get_index_data()
            future_predictions = predict_next_30_days(stock_data)
            plot_stock_data(stock_data, ticker, future_predictions, index_data)
            st.subheader("Company Performance Analysis")
            stock_info = yf.Ticker(ticker).info
            st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
            st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
            st.write(f"- **Net Income:** {format_currency(stock_info.get('netIncome', 0))}")
            st.write(f"- **Earnings Per Share (EPS):** {stock_info.get('trailingEps', 'N/A')}")
            st.write(f"- **Price-to-Earnings (P/E) Ratio:** {stock_info.get('trailingPE', 'N/A')}")
            
            st.subheader("Future Stock Performance Outlook")
            if future_predictions[-1] > stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Buy** - The stock is predicted to rise in value over the next 30 days.")
            elif future_predictions[-1] < stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Sell** - The stock is predicted to decline, consider reducing holdings.")
            else:
                st.write("**Recommendation: Hold** - The stock is expected to remain stable.")
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()
