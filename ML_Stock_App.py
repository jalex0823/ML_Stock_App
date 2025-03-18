import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

PASTEL_COLORS = {
    "Close Price": "#76C7C0",
    "30-Day Forecast": "#FFD580",
    "Background": "#1E1E1E",
    "Grid": "#444444",
    "Text": "#E0E0E0",
}

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
    """Fetches the top 5 performing stocks dynamically based on year-to-date performance."""
    try:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]
        stock_data = []
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="ytd")  
            if len(hist) > 1:
                start_price = hist["Close"].iloc[0]
                current_price = hist["Close"].iloc[-1]
                ytd_change = current_price - start_price
                ytd_percent = (ytd_change / start_price) * 100
            else:
                current_price, ytd_change, ytd_percent = 0, 0, 0  
            
            stock_info = stock.info
            stock_name = stock_info.get("longName", ticker)
            stock_data.append((stock_name, ticker, format_currency(current_price), format_currency(ytd_change), f"{ytd_percent:.2f}%"))
        
        stock_data = sorted(stock_data, key=lambda x: float(x[3].replace("$", "").replace("B", "").replace("M", "").replace("K", "")), reverse=True)[:5]
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

def predict_next_30_days(df):
    """Performs linear regression to predict the next 30 days of stock prices."""
    if df.empty or len(df) < 10:
        return np.array([])
    df['Days'] = np.arange(len(df))
    model = LinearRegression().fit(df[['Days']], df['Close'])
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

def plot_stock_data(df, stock_symbol, future_predictions):
    """Generates a visualization of stock prices with forecast trends using pastel colors."""
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PASTEL_COLORS["Background"])
    ax.set_facecolor(PASTEL_COLORS["Background"])
    
    ax.plot(df.index, df["Close"], label="Close Price", color=PASTEL_COLORS["Close Price"], linewidth=2)
    
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        ax.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color=PASTEL_COLORS["30-Day Forecast"], linewidth=2)

    ax.set_xlabel("Date", color=PASTEL_COLORS["Text"], fontsize=12)
    ax.set_ylabel("Stock Price (USD)", color=PASTEL_COLORS["Text"], fontsize=12)
    ax.tick_params(axis='x', colors=PASTEL_COLORS["Text"])
    ax.tick_params(axis='y', colors=PASTEL_COLORS["Text"])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax.grid(color=PASTEL_COLORS["Grid"], linestyle="dotted")

    legend = ax.legend(loc='upper left', fontsize='small', facecolor=PASTEL_COLORS["Background"], framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color(PASTEL_COLORS["Text"])

    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Option Recommender", page_icon="📊", layout="wide")
    st.title("Stock Option Recommender")
    
    # Display Top 5 Performing Stocks with YTD Performance
    top_stocks = get_top_stocks()
    if top_stocks:
        st.markdown("<h3 style='color:#FFD580;'>Top 5 Performing Stocks</h3>", unsafe_allow_html=True)
        df_top_stocks = pd.DataFrame(top_stocks, columns=["Company Name", "Symbol", "Stock Price", "YTD Change", "YTD % Change"])

        # Add a selectbox to choose a stock from the top list
        selected_stock = st.selectbox("Select a Stock", df_top_stocks["Company Name"], index=None, placeholder="Select a top-performing stock")

        # If user selects a stock, automatically search it
        if selected_stock:
            st.session_state["company_input"] = selected_stock
            st.session_state["trigger_prediction"] = True

    # User Input for Searching Stocks
    company_name = st.text_input("Enter Company Name:", key="company_input")

    # Prediction button
    if st.button("Predict") or st.session_state.get("trigger_prediction", False):
        stock_symbol = get_stock_symbol(company_name)
        st.session_state["trigger_prediction"] = False  # Reset trigger

        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_info = get_stock_info(stock_symbol)
            if stock_data is not None:
                future_predictions = predict_next_30_days(stock_data)
                plot_stock_data(stock_data, stock_symbol, future_predictions)
                
                # Display Stock Details Below Chart
                st.markdown(f"<h3 style='color:#FFD580;'>Stock Details for {stock_symbol.upper()}</h3>", unsafe_allow_html=True)
                last_known_price = stock_data["Close"].iloc[-1]
                final_forecast_price = future_predictions[-1]

                recommendation = "Buy" if final_forecast_price > last_known_price else "Sell"
                reason = "Stock expected to increase, good buying opportunity." if recommendation == "Buy" else "Stock expected to decline, consider selling."

                st.write(f"**Recommendation:** {recommendation} - {reason}")
                st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                st.write(f"**Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
                st.write(f"**Share Price:** {format_currency(stock_info.get('regularMarketPrice', 0))}")

if __name__ == "__main__":
    main()
