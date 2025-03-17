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

def get_top_stock():
    """Fetches the most highly traded stock from S&P 500 dynamically."""
    try:
        sp500 = yf.Ticker("^GSPC")
        stock_info = sp500.info
        top_stock_name = stock_info.get("longName", "S&P 500")
        top_stock_symbol = stock_info.get("symbol", "^GSPC")
        top_stock_price = stock_info.get("regularMarketPrice", 0)
    except Exception:
        top_stock_name, top_stock_symbol, top_stock_price = "S&P 500", "^GSPC", 0
    return top_stock_name, top_stock_symbol, top_stock_price

def get_stock_symbol(company_name):
    """Search for a stock symbol using fuzzy matching on S&P 500 data."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        if "Security" not in table.columns or "Symbol" not in table.columns:
            return None
        company_list = table[['Security', 'Symbol']].dropna()
        company_list['Security'] = company_list['Security'].str.lower()
        result = process.extractOne(company_name.lower(), company_list['Security'])
        if result and result[1] >= 70:
            return company_list.loc[company_list['Security'] == result[0], 'Symbol'].values[0]
        return None
    except Exception:
        return None

def get_stock_data(stock_symbol):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    return hist if not hist.empty else None

def get_stock_info(stock_symbol):
    """Fetches stock information like market cap and revenue."""
    stock = yf.Ticker(stock_symbol)
    return stock.info

def get_index_data():
    """Fetches historical data for major stock indices using a single request."""
    indices = ["^GSPC", "^IXIC", "^DJI"]
    return yf.download(indices, period="1y")

def predict_next_30_days(df):
    """Performs linear regression to predict the next 30 days of stock prices."""
    if df is None or df.empty or len(df) < 10:
        return np.array([])
    df['Days'] = np.arange(len(df))
    model = LinearRegression().fit(df[['Days']], df['Close'])
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

def plot_stock_data(df, stock_symbol, future_predictions, index_data):
    """Generates a visualization of stock prices and index trends with a modern theme."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor('black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    ax1.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=3)
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        ax1.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="lime", linewidth=3)
    for name in ["^GSPC", "^IXIC", "^DJI"]:
        if name in index_data.columns:
            ax2.plot(index_data.index, index_data[("Close", name)], linestyle="dotted", label=name, linewidth=2)
    ax1.set_xlabel("Date", color='white')
    ax1.set_ylabel("Stock Price (USD)", color='white')
    ax2.set_ylabel("Index Values", color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax1.grid(color='gray', linestyle='dotted')
    legend = ax1.legend(loc='upper center', fontsize='medium', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    top_stock_name, top_stock_symbol, top_stock_price = get_top_stock()
    st.markdown(f"<h3 style='color:white;'>Top Performing Stock: {top_stock_name} - {format_currency(top_stock_price)}</h3>", unsafe_allow_html=True)
    company_name = st.text_input("Enter Company Name:")
    if st.button("Predict"):
        stock_symbol = get_stock_symbol(company_name)
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_info = get_stock_info(stock_symbol)
            if stock_data is not None:
                index_data = get_index_data()
                future_predictions = predict_next_30_days(stock_data)
                plot_stock_data(stock_data, stock_symbol, future_predictions, index_data)
                st.markdown(f"<h3 style='color:white;'>Stock Details for {stock_symbol.upper()}</h3>", unsafe_allow_html=True)
                recommendation = "Buy" if future_predictions[-1] > stock_data['Close'].iloc[-1] else "Sell"
                st.write(f"**Recommendation:** {recommendation} - {'Stock expected to increase, good buying opportunity' if recommendation == 'Buy' else 'Stock expected to decline, consider selling.'}")
                st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                st.write(f"**Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
                st.write(f"**Share Price:** {format_currency(stock_info.get('regularMarketPrice', 0))}")
            else:
                st.error("No data available for the selected company.")
        else:
            st.error("Unable to find stock symbol for the given company.")
if __name__ == "__main__":
    main()
