import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fuzzywuzzy import process
from sklearn.linear_model import LinearRegression

# Initialize session state
if "selected_stock" not in st.session_state:
    st.session_state["selected_stock"] = "AAPL"
if "search_input" not in st.session_state:
    st.session_state["search_input"] = ""

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0F172A; font-family: 'Arial', sans-serif; }
    .stock-btn { width: 100%; height: 100px; font-size: 16px; text-align: center; 
                 background: #1E40AF; color: white; border-radius: 5px; border: none;
                 transition: 0.3s; cursor: pointer; padding: 15px; display: flex;
                 flex-direction: column; align-items: center; justify-content: center; }
    .stock-btn:hover { background: #3B82F6; transform: scale(1.05); }
    .info-box { font-size: 16px; padding: 10px; background: #1E293B; color: white; 
                border-radius: 5px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)
# Load S&P 500 list
@st.cache_data
def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        table = pd.read_html(url, header=0)[0]
        return table[['Security', 'Symbol']].dropna()
    except:
        return pd.DataFrame(columns=['Security', 'Symbol'])

sp500_list = get_sp500_list()

# Convert search input to stock symbol
def get_stock_symbol(search_input):
    search_input = search_input.strip().upper()
    if search_input in sp500_list['Symbol'].values:
        return search_input
    result = process.extractOne(search_input, sp500_list['Security'])
    if result and result[1] >= 70:
        return sp500_list.loc[sp500_list['Security'] == result[0], 'Symbol'].values[0]
    return None

# Get top 15 performing stocks dynamically
def get_top_stocks():
    tickers = sp500_list['Symbol'].tolist()[:50]  # Top 50 for better sampling
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            price = info.get("regularMarketPrice", 0)
            change = info.get("52WeekChange", 0)
            delta = price * change
            data.append({
                "symbol": t,
                "name": info.get("shortName", t),
                "price": price,
                "change": delta,
                "percent": change
            })
        except:
            continue
    sorted_data = sorted(data, key=lambda x: x["percent"], reverse=True)
    return sorted_data[:15]

# Search bar
st.markdown("<h3 style='color:white;'>🔍 Search by Company Name or Symbol</h3>", unsafe_allow_html=True)
search_input = st.text_input("", value=st.session_state["search_input"], placeholder="Type stock symbol or company name...").strip().upper()

# Top 15 Buttons in 3 columns
st.markdown("<h3 style='color:white;'>📈 Top 15 Performing Stocks</h3>", unsafe_allow_html=True)
top_stocks = get_top_stocks()

col1, col2, col3 = st.columns(3)
for i, stock in enumerate(top_stocks):
    col = [col1, col2, col3][i % 3]
    with col:
        btn_label = f"**{stock['name']}**  \n{stock['symbol']}  \n💲{stock['price']:.2f}  \n📈 {stock['change']:+.2f} ({stock['percent']:.2%})"
        if st.button(btn_label, key=f"top_{i}", use_container_width=True):
            st.session_state["search_input"] = ""
            st.session_state["selected_stock"] = stock["symbol"]

# Determine which stock is selected
selected_stock = get_stock_symbol(search_input) if search_input else st.session_state["selected_stock"]

if not selected_stock:
    st.error("⚠️ Invalid company name or symbol. Please try again.")
    st.stop()