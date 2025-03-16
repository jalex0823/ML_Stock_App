import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

# Function to format large numbers (e.g., Thousands, Millions, Billions)
def format_currency(value):
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"

# Function to fetch the most highly traded stock of the day
def get_most_traded_stock():
    url = "https://finance.yahoo.com/most-active"
    df = pd.read_html(url)[0]  # Read the first table
    most_traded = df.iloc[0]  # Select the top most active stock

    ticker = most_traded["Symbol"]
    company_name = most_traded["Name"]
    volume = format_currency(most_traded["Volume"])
    last_price = format_currency(most_traded["Last Price"])
    share_price = format_currency(most_traded["Last Price"])  # Using Last Price as Share Price

    return ticker, company_name, volume, last_price, share_price

# Function to get stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1y")

# Function to predict next 30 days using Linear Regression
def predict_next_30_days(df):
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    return model.predict(future_days)

# Function to plot stock and index data
def plot_stock_data(df, company_name, future_predictions):
    st.subheader(f"Stock Price Chart for {company_name}")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')

    # Plot actual stock data and moving averages
    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    ax1.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=2)

    # Set labels and grid
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Stock Price (USD)", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.grid(color='gray', linestyle='dotted')

    # Legend
    legend = ax1.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    st.pyplot(fig)

# Main function
def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")

    # Fetch Most Highly Traded Stock of the Day
    try:
        top_ticker, top_name, top_volume, top_last_price, top_share_price = get_most_traded_stock()
        st.subheader(f"📈 Most Traded Stock Today: {top_name} ({top_ticker})")
        st.write(f"- **Last Price:** {top_last_price}")
        st.write(f"- **Trading Volume:** {top_volume}")
        st.write(f"- **Share Price:** {top_share_price}")
    except:
        st.write("⚠️ Unable to fetch the most traded stock at the moment.")

    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL):")

    if st.button("Predict"):
        if stock_symbol:
            try:
                stock_info = yf.Ticker(stock_symbol).info
                company_name = stock_info.get("longName", stock_symbol.upper())
                last_price = format_currency(stock_info.get("previousClose", 0))
                share_price = format_currency(stock_info.get("currentPrice", 0))

                stock_data = get_stock_data(stock_symbol)
                future_predictions = predict_next_30_days(stock_data)

                plot_stock_data(stock_data, company_name, future_predictions)

                # **Company Performance Section**
                st.subheader(f"{company_name} ({stock_symbol.upper()}) Company Performance")
                st.write(f"- **Last Price:** {last_price}")
                st.write(f"- **Share Price:** {share_price}")
                st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
                st.write(f"- **Net Income:** {format_currency(stock_info.get('netIncome', 0))}")
                st.write(f"- **Earnings Per Share (EPS):** {stock_info.get('trailingEps', 'N/A')}")
                st.write(f"- **Price-to-Earnings (P/E) Ratio:** {stock_info.get('trailingPE', 'N/A')}")

                # **Prediction Recommendation**
                st.subheader(f"Future Outlook for {company_name} ({stock_symbol.upper()})")
                if future_predictions[-1] > stock_data["Close"].iloc[-1]:
                    st.write(f"**Recommendation: Buy** - {company_name} is predicted to rise in value over the next 30 days.")
                elif future_predictions[-1] < stock_data["Close"].iloc[-1]:
                    st.write(f"**Recommendation: Sell** - {company_name} is predicted to decline, consider reducing holdings.")
                else:
                    st.write(f"**Recommendation: Hold** - {company_name} is expected to remain stable.")
            except Exception as e:
                st.error(f"Failed to fetch data for {stock_symbol.upper()}. Please check the ticker and try again.")
        else:
            st.error("Please enter a valid stock ticker.")

if __name__ == "__main__":
    main()