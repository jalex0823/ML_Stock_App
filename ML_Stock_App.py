import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
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

def get_top_stocks():
    """Fetches the top 5 performing stocks dynamically based on YTD performance."""
    try:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]
        stock_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            stock_name = stock_info.get("longName", ticker)
            stock_price = stock_info.get("regularMarketPrice", 0)
            ytd_change = stock_info.get("ytdReturn", 0) * 100  # Convert to percentage
            ytd_dollar = stock_info.get("fiftyTwoWeekHigh", 0) - stock_info.get("fiftyTwoWeekLow", 0)
            stock_data.append((stock_name, ticker, stock_price, format_currency(ytd_dollar), f"{ytd_change:.2f}%"))
        
        stock_data = sorted(stock_data, key=lambda x: x[2], reverse=True)[:5]
        return stock_data
    except Exception:
        return []

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

def add_technical_indicators(df):
    """Calculates key technical indicators."""
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_signal'] = macd.iloc[:, 1]
    bollinger = ta.bbands(df['Close'], length=20)
    if bollinger is not None:
        df['Bollinger_Upper'] = bollinger.iloc[:, 0]
        df['Bollinger_Middle'] = bollinger.iloc[:, 1]
        df['Bollinger_Lower'] = bollinger.iloc[:, 2]
    df.dropna(inplace=True)
    return df

def predict_next_30_days(df):
    """Performs linear regression to predict the next 30 days of stock prices."""
    if df.empty or len(df) < 10:
        return np.array([])
    df['Days'] = np.arange(len(df))
    model = LinearRegression().fit(df[['Days']], df['Close'])
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

def plot_stock_data(df, stock_symbol, future_predictions):
    """Generates a visualization of stock prices with forecast trends."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        ax.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color="orange", linewidth=2)

    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Stock Price (USD)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax.grid(color='gray', linestyle='dotted')

    legend = ax.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    
    # Display Top 5 Performing Stocks
    top_stocks = get_top_stocks()
    if top_stocks:
        st.markdown("<h3 style='color:white;'>Top 5 Performing Stocks</h3>", unsafe_allow_html=True)
        df_top_stocks = pd.DataFrame(top_stocks, columns=["Company Name", "Symbol", "Price", "YTD Change ($)", "YTD Change (%)"])
        st.dataframe(df_top_stocks.style.set_properties(**{'background-color': 'black', 'color': 'white'}))

    # **Dropdown and Input Box Below Table**
    selected_stock = st.selectbox("Select a Stock", options=[""] + [stock[0] for stock in top_stocks], key="dropdown_select")
    company_name = st.text_input("Or Enter a Company Name:", value="", key="company_input")

    # **Ensure both options work independently**
    if selected_stock:
        company_name = selected_stock

    if st.button("Predict"):
        stock_symbol = get_stock_symbol(company_name)
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_info = get_stock_info(stock_symbol)
            if stock_data is not None:
                stock_data = add_technical_indicators(stock_data)
                future_predictions = predict_next_30_days(stock_data)
                plot_stock_data(stock_data, stock_symbol, future_predictions)

                # **Fix Recommendation Based on Trend**
                last_known_price = stock_data["Close"].iloc[-1]
                if future_predictions.size > 0:
                    avg_forecast_price = np.mean(future_predictions)
                    if avg_forecast_price > last_known_price:
                        recommendation = "Buy"
                        reason = "Stock expected to increase, good buying opportunity."
                    else:
                        recommendation = "Sell"
                        reason = "Stock expected to decline, consider selling."
                    st.markdown(f"<h3 style='color:white;'>Stock Details for {stock_symbol.upper()}</h3>", unsafe_allow_html=True)
                    st.write(f"**Recommendation:** {recommendation} - {reason}")
                    st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                    st.write(f"**Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
                    st.write(f"**Share Price:** {format_currency(stock_info.get('regularMarketPrice', 0))}")
            else:
                st.error("No data available for the selected company.")
        else:
            st.error("Unable to find stock symbol for the given company.")

if __name__ == "__main__":
    main()
