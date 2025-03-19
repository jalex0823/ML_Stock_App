import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from fuzzywuzzy import process

# 🎨 Theme Colors
BACKGROUND_COLOR = "#1E3A8A"  # Deep Blue
TEXT_COLOR = "#E5E7EB"  # Light Gray
HIGHLIGHT_COLOR = "#34D399"  # Green for positive
NEGATIVE_COLOR = "#EF4444"  # Red for negative

# ✅ Apply Custom Theme
st.set_page_config(page_title="Stock Option Recommender", page_icon="📈", layout="wide")

# 📌 Set Background Color for the Full Page
page_bg = f"""
<style>
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .stDataFrame {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 📌 Format currency
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

# 🔥 Get Top 5 Stocks Based on YTD Change
def get_top_stocks():
    """Fetches top 5 performing stocks dynamically based on year-to-date performance."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]
    stock_data = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="ytd")  
        if hist.empty:
            continue

        ytd_start_price = hist.iloc[0]["Open"]
        current_price = hist.iloc[-1]["Close"]
        ytd_change = current_price - ytd_start_price
        ytd_percent = (ytd_change / ytd_start_price) * 100

        stock_info = stock.info
        stock_name = stock_info.get("longName", ticker)
        
        stock_data.append((stock_name, ticker, format_currency(current_price), format_currency(ytd_change), f"{ytd_percent:.2f}%"))

    stock_data = sorted(stock_data, key=lambda x: float(x[3].replace("$", "").replace("B", "").replace("M", "").replace("K", "")), reverse=True)[:5]
    return stock_data

# 🎯 Get Stock Symbol from Company Name
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

# 📊 Fetch Stock Data
def get_stock_data(stock_symbol):
    """Fetches historical stock data from Yahoo Finance."""
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period="1y")
    return hist if not hist.empty else None

# 📌 Fetch Stock Info
def get_stock_info(stock_symbol):
    """Fetches stock information like market cap and revenue."""
    stock = yf.Ticker(stock_symbol)
    return stock.info

# 🔮 Predict Next 30 Days
def predict_next_30_days(df):
    """Performs linear regression to predict the next 30 days of stock prices."""
    if df.empty or len(df) < 10:
        return np.array([])
    df['Days'] = np.arange(len(df))
    model = LinearRegression().fit(df[['Days']], df['Close'])
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

# 📈 Plot Stock Data
def plot_stock_data(df, stock_symbol, future_predictions):
    """Generates a visualization of stock prices with forecast trends."""
    st.subheader(f"📊 {stock_symbol.upper()} Stock Trends")

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    ax.plot(df.index, df["Close"], label="Close Price", color="cyan", linewidth=2)
    
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        ax.plot(future_dates, future_predictions, label="30-Day Forecast", linestyle="dashed", color=HIGHLIGHT_COLOR, linewidth=2)

    ax.set_xlabel("Date", color=TEXT_COLOR)
    ax.set_ylabel("Stock Price (USD)", color=TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.tick_params(axis='y', colors=TEXT_COLOR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_currency(x)))
    ax.grid(color='gray', linestyle='dotted')

    legend = ax.legend(loc='upper left', fontsize='small', facecolor='black', framealpha=0.9, edgecolor='white')
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    st.pyplot(fig)

# 🎯 Main Function
def main():
    st.title("💹 Stock Option Recommender")

    # 📌 Display Top 5 Performing Stocks
    top_stocks = get_top_stocks()
    if top_stocks:
        st.markdown("<h3 style='color:white;'>Top 5 Performing Stocks</h3>", unsafe_allow_html=True)
        df_top_stocks = pd.DataFrame(top_stocks, columns=["Company Name", "Symbol", "Price", "YTD Change ($)", "YTD Change (%)"])
        st.dataframe(df_top_stocks.style.set_properties(**{'background-color': BACKGROUND_COLOR, 'color': TEXT_COLOR}))

    # 📌 Dropdown for Stock Selection
    selected_stock = st.selectbox("🔍 Select a Stock:", [""] + [row[0] for row in top_stocks], key="stock_dropdown")

    # 📌 Search Box
    company_name = st.text_input("Or Enter a Company Name:", value=st.session_state.get("search_box", ""))

    # ✅ Ensure dropdown and search box properly clear each other
    if selected_stock and selected_stock != company_name:
        st.session_state["search_box"] = selected_stock  
    elif company_name and company_name != selected_stock:
        st.session_state["stock_dropdown"] = ""  

    if st.button("📊 Predict"):
        stock_symbol = get_stock_symbol(company_name)
        if stock_symbol:
            stock_data = get_stock_data(stock_symbol)
            stock_info = get_stock_info(stock_symbol)
            if stock_data is not None:
                future_predictions = predict_next_30_days(stock_data)
                plot_stock_data(stock_data, stock_symbol, future_predictions)
                
                # 📌 Recommendation Logic
                recommendation = "Buy" if future_predictions[-1] > stock_data["Close"].iloc[-1] else "Sell"
                st.markdown(f"<h3 style='color:{TEXT_COLOR};'>Stock Details for {stock_symbol.upper()}</h3>", unsafe_allow_html=True)
                st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                st.write(f"**Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
                st.write(f"**Recommendation:** {recommendation}")

if __name__ == "__main__":
    main()
