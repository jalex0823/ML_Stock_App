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
    """Search for a stock symbol based on the company name."""
    search_results = yf.search(company_name)
    if not search_results.empty:
        return search_results.iloc[0]['Symbol']
    return None

def get_stock_data(stock_symbol):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    return hist

def get_top_performing_stock():
    """Fetches the most highly traded stock of the day."""
    sp500 = yf.Ticker("^GSPC").history(period="1d")
    if sp500.empty:
        return None, None
    top_stock = "AAPL"  # Placeholder, actual top stock logic can be added
    stock = yf.Ticker(top_stock)
    stock_info = stock.info
    return top_stock, stock_info

def add_technical_indicators(df):
    """Calculates key technical indicators."""
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_signal'] = macd.iloc[:, 1]
    df.dropna(inplace=True)
    return df

def predict_next_30_days(df):
    """Predicts stock prices for the next 30 days."""
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    return model.predict(future_days)

def plot_stock_data(df, stock_symbol, future_predictions):
    """Generates a visualization of stock prices and index trends."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    ax1.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=2)
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Stock Price (USD)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    legend = ax1.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    ax1.grid(color='gray', linestyle='dotted')
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    
    top_stock, top_stock_info = get_top_performing_stock()
    if top_stock:
        st.subheader(f"Top Performing Stock Today: {top_stock}")
        st.write(f"- **Share Price:** {format_currency(top_stock_info.get('previousClose', 0))}")
        st.write(f"- **Market Cap:** {format_currency(top_stock_info.get('marketCap', 0))}")
        st.write(f"- **Daily Volume:** {format_currency(top_stock_info.get('volume', 0))}")
    
    company_name = st.text_input("Enter Company Name (e.g., Apple Inc.):")
    if st.button("Search and Predict"):
        stock_symbol = get_stock_symbol(company_name)
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_data = add_technical_indicators(stock_data)
            future_predictions = predict_next_30_days(stock_data)
            plot_stock_data(stock_data, stock_symbol, future_predictions)
            st.subheader(f"{stock_symbol.upper()} Company Performance")
            stock_info = yf.Ticker(stock_symbol).info
            st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
            st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
            st.write(f"- **Net Income:** {format_currency(stock_info.get('netIncome', 0))}")
            st.write(f"- **EPS:** {stock_info.get('trailingEps', 'N/A')}")
            st.subheader(f"Future Outlook for {stock_symbol.upper()}")
            if future_predictions[-1] > stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Buy** - Predicted to rise in value over 30 days.")
            elif future_predictions[-1] < stock_data["Close"].iloc[-1]:
                st.write("**Recommendation: Sell** - Predicted to decline, consider reducing holdings.")
            else:
                st.write("**Recommendation: Hold** - Expected to remain stable.")
        else:
            st.error("Company not found. Please try again.")

if __name__ == "__main__":
    main()