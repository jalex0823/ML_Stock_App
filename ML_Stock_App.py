import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replacing btalib with pandas_ta
import streamlit as st
import matplotlib.pyplot as plt
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
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bollinger = ta.bbands(df['Close'], length=20)
    df['Bollinger_Upper'] = bollinger['BBU_20_2.0']
    df['Bollinger_Middle'] = bollinger['BBM_20_2.0']
    df['Bollinger_Lower'] = bollinger['BBL_20_2.0']
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
    st.subheader("Stock Price Visualization and Market Index Trends")
    
    st.markdown("""
    **Chart Explanation:**
    - **Blue Line (Close Price):** Represents the daily closing prices of the stock, reflecting market sentiment and investor behavior.
    - **Orange Dashed Line (50-day SMA):** Shows short-term price trends by averaging the last 50 days of closing prices.
    - **Red Dashed Line (200-day SMA):** Represents a long-term market trend, smoothing price fluctuations over 200 days.
    - **Green Dashed Line (30-Day Forecast):** Predicts the stock's future movement based on historical trends and performance indicators.
    
    **Prediction Line Interpretation:**
    - If the **forecasted line trends upwards**, it suggests potential price appreciation, driven by positive momentum and potential company growth.
    - A **flat or declining forecast** indicates weak momentum, suggesting limited growth or even a downturn based on past performance.
    - **Macroeconomic factors, earnings reports, and industry trends** should also be considered alongside this forecast for a well-rounded investment decision.
    """)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Close Price", color="blue")
    ax.plot(df.index, df["SMA_50"], label="50-day SMA", linestyle="dashed", color="orange")
    ax.plot(df.index, df["SMA_200"], label="200-day SMA", linestyle="dashed", color="red")
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    ax.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="green")
    
    for name, data in index_data.items():
        ax.plot(data.index, data["Close"], linestyle="dotted", label=name)
    
    ax.set_title(f"{ticker} Stock Price with 30-Day Forecast and Major Indexes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid()
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
            st.write("The forecast line provides an indication of possible stock movements. However, investors should consider factors such as **market trends, economic conditions, and recent earnings reports** to make informed decisions.")
            st.write("It is recommended to **analyze additional fundamental data, industry trends, and broader market indicators** before making any investment decisions.")
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()
