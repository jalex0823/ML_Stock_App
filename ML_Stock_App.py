import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# 🎨 MSN Watchlist-Themed UI
st.set_page_config(page_title="Stock Watchlist", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #0F172A; color: white;}
    .stDataFrame {background-color: #1E293B !important; color: white;}
    .stButton>button {background-color: #0078D4 !important; color: white !important;}
    .stTextInput>div>div>input {background-color: #1E293B !important; color: white !important; padding: 10px;}
    .stock-card {background-color: #1E293B; padding: 10px; border-radius: 5px; margin-bottom: 8px;}
    .stock-title {color: white; font-size: 16px; font-weight: bold;}
    .stock-metrics {color: #94A3B8; font-size: 14px;}
    .positive {color: #16A34A; font-weight: bold;}
    .negative {color: #DC2626; font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True
)

# 🔥 Format currency display
def format_currency(value):
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"

# 🔍 Get top 5 performing stocks dynamically (based on % change)
def get_top_stocks():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "V", "JNJ"]
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        stock_price = stock_info.get("regularMarketPrice", 0)
        ytd_change = stock_info.get("52WeekChange", 0)
        ytd_change_amt = stock_price * ytd_change
        stock_data.append((stock_info.get("longName", ticker), ticker, stock_price, ytd_change_amt, ytd_change))
    
    stock_data = sorted(stock_data, key=lambda x: x[4], reverse=True)[:5]
    return stock_data

# 🔄 Get stock symbol from company name
def get_stock_symbol(company_name):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        company_list = table[['Security', 'Symbol']].dropna()
        company_list['Security'] = company_list['Security'].str.lower()
        result = process.extractOne(company_name.lower(), company_list['Security'])
        return company_list.loc[company_list['Security'] == result[0], 'Symbol'].values[0] if result and result[1] >= 70 else None
    except Exception:
        return None

# 📊 Get stock historical data based on time filters
def get_stock_data(stock_symbol, period="1y"):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period=period)
    return hist if not hist.empty else None

# 📈 Predict next 30 days using Linear Regression
def predict_next_30_days(df):
    if df.empty or len(df) < 10:
        return np.array([])
    df['Days'] = np.arange(len(df))
    model = LinearRegression().fit(df[['Days']], df['Close'])
    return model.predict(np.arange(len(df), len(df) + 30).reshape(-1, 1))

# 📊 Plot stock chart with forecast
def plot_stock_data(df, stock_symbol, future_predictions):
    st.subheader(f"{stock_symbol.upper()} Stock Price & Market Trends")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name="Close Price", line=dict(color="cyan", width=2)))
    
    future_dates = pd.date_range(start=df.index[-1], periods=30, freq='D')
    if future_predictions.size > 0:
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode="lines", name="30-Day Forecast", line=dict(color="orange", width=2, dash="dot")))

    fig.update_layout(
        title=f"{stock_symbol.upper()} Stock Price & Trends",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        font=dict(color="white"),
        legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# 🌟 MAIN APP LAYOUT
def main():
    col1, col2 = st.columns([1, 3])  # Sidebar (1) | Main Content (3)

    # 📌 Sidebar (Top 5 Stocks Watchlist)
    with col1:
        st.markdown("<h3 style='color:white;'>Top 5 Stocks</h3>", unsafe_allow_html=True)
        top_stocks = get_top_stocks()

        for name, symbol, price, ytd_amt, ytd_pct in top_stocks:
            color = "green" if ytd_pct > 0 else "red"
            hist = get_stock_data(symbol, "6mo")
            if hist is not None and not hist.empty:
                sparkline = go.Figure()
                sparkline.add_trace(go.Scatter(y=hist["Close"][-20:], mode='lines', line=dict(color='white', width=2)))
                sparkline.update_layout(
                    paper_bgcolor="#1E293B", plot_bgcolor="#1E293B",
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False)
                )

                st.markdown(
                    f"<div class='stock-card' onclick=\"document.getElementById('stock_input').value='{name}';\">"
                    f"<strong class='stock-title'>{name} ({symbol})</strong><br>"
                    f"<span class='stock-metrics'>Price: {format_currency(price)}</span><br>"
                    f"<span style='color:{color}; font-weight:bold;'>YTD: {format_currency(ytd_amt)} ({ytd_pct:.2%})</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                st.plotly_chart(sparkline, use_container_width=True)

    # 📌 Main Section (Graph & Details)
    with col2:
        # Time Range Selection (Like MSN Watchlist)
        period = st.selectbox("Select Time Range:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"])

        stock_input = st.text_input("Enter Company Name:", key="stock_input")
        if st.button("Search"):
            stock_symbol = get_stock_symbol(stock_input)
            if stock_symbol:
                stock_data = get_stock_data(stock_symbol, period)
                if stock_data is not None:
                    future_predictions = predict_next_30_days(stock_data)
                    plot_stock_data(stock_data, stock_symbol, future_predictions)
                    
                    stock_info = yf.Ticker(stock_symbol).info
                    st.markdown(f"<h3 style='color:white;'>Stock Details for {stock_symbol.upper()}</h3>", unsafe_allow_html=True)
                    st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                    st.write(f"**Revenue:** {format_currency(stock_info.get('totalRevenue', 0))}")
                    st.write(f"**Share Price:** {format_currency(stock_info.get('regularMarketPrice', 0))}")

if __name__ == "__main__":
    main()
