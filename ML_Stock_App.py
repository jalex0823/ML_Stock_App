import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

def get_stock_symbol(company_name):
    search_results = yf.Ticker(company_name).history(period="1d")
    if search_results.empty:
        return None
    return company_name

def get_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    return hist if not hist.empty else None

def get_index_data():
    indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
    index_data = {name: yf.Ticker(symbol).history(period="1y") for name, symbol in indices.items()}
    return index_data

def add_technical_indicators(df):
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_signal'] = macd.iloc[:, 1]
    df.dropna(inplace=True)
    return df

def predict_next_30_days(df):
    if len(df) < 10:
        return None  # Not enough data points
    df['Days'] = np.arange(len(df))
    X = df[['Days']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
    return model.predict(future_days)

def plot_stock_data(df, stock_symbol, future_predictions, index_data):
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
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
    company_name = st.text_input("Enter Company Name (e.g., Apple):")
    if st.button("Predict"):
        stock_symbol = get_stock_symbol(company_name)
        if not stock_symbol:
            st.error("Invalid company name. Please try again.")
            return
        stock_data = get_stock_data(stock_symbol)
        if stock_data is None:
            st.error("No stock data found for this company.")
            return
        stock_data = add_technical_indicators(stock_data)
        index_data = get_index_data()
        future_predictions = predict_next_30_days(stock_data)
        plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
        st.subheader(f"{stock_symbol.upper()} Company Performance")
        stock_info = yf.Ticker(stock_symbol).info
        st.write(f"- **Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
        st.write(f"- **Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
        st.write(f"- **Net Income:** {format_currency(stock_info.get('netIncome', 0))}")
        st.write(f"- **Earnings Per Share (EPS):** {stock_info.get('trailingEps', 'N/A')}")
        st.write(f"- **Price-to-Earnings (P/E) Ratio:** {stock_info.get('trailingPE', 'N/A')}")
        st.subheader(f"Future Outlook for {stock_symbol.upper()}")
        if future_predictions is not None and future_predictions[-1] > stock_data["Close"].iloc[-1]:
            st.write("**Recommendation: Buy** - The stock is predicted to rise in value over the next 30 days.")
        elif future_predictions is not None and future_predictions[-1] < stock_data["Close"].iloc[-1]:
            st.write("**Recommendation: Sell** - The stock is predicted to decline, consider reducing holdings.")
        else:
            st.write("**Recommendation: Hold** - The stock is expected to remain stable.")

if __name__ == "__main__":
    main()