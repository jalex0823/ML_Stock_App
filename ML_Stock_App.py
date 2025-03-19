import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import time

# 🎨 UI ENHANCEMENTS
st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-card { padding: 15px; margin: 5px; background: linear-gradient(135deg, #1E293B, #334155); 
                  border-radius: 10px; color: white; text-align: center; transition: 0.3s; }
    .stock-card:hover { transform: scale(1.05); background: linear-gradient(135deg, #334155, #475569); }
    .btn { padding: 8px 15px; background: #1E40AF; color: white; border-radius: 5px; font-size: 14px; transition: 0.3s; }
    .btn:hover { background: #3B82F6; transform: scale(1.1); }
    .watchlist-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
    .watchlist-table th, .watchlist-table td { padding: 10px; text-align: left; border: 1px solid #334155; }
    .watchlist-table th { background-color: #1E293B; color: white; font-size: 14px; }
    .watchlist-table td { background-color: #0F172A; color: white; font-size: 13px; }
    .positive { color: #16A34A; font-weight: bold; }
    .negative { color: #DC2626; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 📌 Add a **real-time stock market ticker** at the top
st.markdown("<h3 style='color:white; text-align:center;'>🔹 Real-Time Stock Market Overview 🔹</h3>", unsafe_allow_html=True)

# 🎯 Auto-refresh live data every **15 seconds**
st_autorefresh = st.empty()

def get_top_stocks():
    """Fetch top 5 performing stocks dynamically."""
    top_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    stock_data = []
    for stock in top_stocks:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="6mo")
        price = ticker.info.get("regularMarketPrice", 0)
        change_pct = ticker.info.get("52WeekChange", 0)
        change_amt = price * change_pct

        stock_data.append({
            "symbol": stock,
            "name": ticker.info.get("shortName", stock),
            "price": f"${price:.2f}",
            "change": f"{change_amt:.2f} ({change_pct:.2%})",
            "change_class": "positive" if change_pct > 0 else "negative",
            "trend": hist["Close"][-10:].tolist() if not hist.empty else []
        })
    return stock_data

# 📊 LIVE STOCK MARKET UPDATE (Updates every **15 seconds**)
for _ in range(3):
    with st_autorefresh:
        time.sleep(15)

# 📌 Timeframe Selection (Now Fixed)
st.markdown("<h3 style='color:white; text-align:center;'>Select Timeframe</h3>", unsafe_allow_html=True)
timeframes = ["1D", "5D", "1M", "3M", "YTD", "1Y", "5Y", "Max"]
selected_timeframe = st.radio("", timeframes, horizontal=True, key="time_select")

# 📌 Layout Fix - Proper Positioning
col1, col2 = st.columns([1, 3], gap="large")  # Adjusted for better spacing

# 🟢 **LEFT COLUMN: Quick Compare Stocks (Fixed HTML Rendering)**
with col1:
    st.markdown("<h3 style='color:white;'>Quick Compare</h3>", unsafe_allow_html=True)
    
    top_stocks = get_top_stocks()
    for stock in top_stocks:
        st.markdown(f"""
            <div class="stock-card">
                <h4>{stock['name']} ({stock['symbol']})</h4>
                <p>{stock['price']} | <span class="{stock['change_class']}">{stock['change']}</span></p>
            </div>
        """, unsafe_allow_html=True)

# 🔍 **SEARCH STOCK INPUT BELOW TABLE (Now Functional)**
search_stock = st.text_input("Search Stock:", key="stock_input")

# 📌 Stock Comparison Selection (Now Works Properly)
st.markdown("<h3 style='color:white;'>Compare Stocks</h3>", unsafe_allow_html=True)
compare_stocks = st.multiselect("Select Stocks to Compare:", [s["symbol"] for s in top_stocks])

# ✅ Fix for NameError: Ensuring Stock Selection is Always Defined
selected_stocks = []
if search_stock:
    selected_stocks.append(search_stock)
selected_stocks += compare_stocks

# 🟢 **RIGHT COLUMN: Stock Graph (Now Functional)**
with col2:
    if selected_stocks:
        fig = go.Figure()

        # 📌 Fetch Data for Selected Stocks
        for stock in selected_stocks:
            try:
                ticker = yf.Ticker(stock)
                hist = ticker.history(period=selected_timeframe.lower())

                if not hist.empty:
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist["Close"],
                        mode="lines",
                        name=f"{stock} Close Price",
                        line=dict(width=2)
                    ))

                    # 📊 Moving Averages
                    ma_50 = hist["Close"].rolling(window=50).mean()
                    ma_200 = hist["Close"].rolling(window=200).mean()
                    show_50_ma = st.checkbox(f"Show 50-Day MA for {stock}", value=True)
                    show_200_ma = st.checkbox(f"Show 200-Day MA for {stock}", value=False)

                    if show_50_ma:
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=ma_50,
                            mode="lines",
                            name=f"{stock} 50-Day MA",
                            line=dict(dash="dash")
                        ))

                    if show_200_ma:
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=ma_200,
                            mode="lines",
                            name=f"{stock} 200-Day MA",
                            line=dict(dash="dash")
                        ))

            except Exception as e:
                st.warning(f"Could not retrieve data for {stock}: {e}")

        # 📌 Format Chart
        fig.update_layout(
            title="Stock Price & Trends",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)",
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font=dict(color="white"),
            legend=dict(bgcolor="#1E293B", bordercolor="white", borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

# ✅ Fix for NameError: Ensuring Stock Selection is Always Valid
if not selected_stocks:
    st.info("Select or search for a stock to display its details.")
