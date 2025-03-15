import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta  # Replacing btalib with pandas_ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Fetch Stock and Options Data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def get_options_data(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        print("No options data available for this stock.")
        return pd.DataFrame()
    options_data = []
    for exp in expirations[:3]:  # Limit to 3 expirations to reduce API calls
        opt_chain = stock.option_chain(exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        options_data.append(pd.concat([calls, puts]))
    return pd.concat(options_data) if options_data else pd.DataFrame()

# Step 2: Compute Technical Indicators
def add_technical_indicators(df):
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bollinger = ta.bbands(df['Close'], length=20)
    df['Bollinger_Upper'] = bollinger['BBU_20_2.0']
    df['Bollinger_Middle'] = bollinger['BBM_20_2.0']
    df['Bollinger_Lower'] = bollinger['BBL_20_2.0']
    df.dropna(inplace=True)
    return df

# Step 3: Prepare ML Dataset
def prepare_ml_dataset(df):
    df['Future_Return'] = df['Close'].pct_change().shift(-1)  # Predict next-day return
    df.dropna(inplace=True)
    df['Label'] = np.where(df['Future_Return'] > 0.01, 1, 0)  # Buy if expected return > 1%
    features = ['SMA_50', 'SMA_200', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    if df[features].isnull().sum().sum() > 0:
        print("Warning: Missing values detected in features. Dropping NaNs.")
        df.dropna(inplace=True)
    X = df[features]
    y = df['Label']
    return train_test_split(X, y, test_size=0.2, random_state=42) if not X.empty else (None, None, None, None)

# Step 4: Train Model
def train_model(X_train, y_train):
    if X_train is None or y_train is None:
        print("Insufficient data for training.")
        return None
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
    if model is None or X_test is None or y_test is None:
        return "N/A"
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

# Main Execution
if __name__ == "__main__":
    try:
        ticker = "AAPL"  # Example
        stock_data = get_stock_data(ticker)
        stock_data = add_technical_indicators(stock_data)
        X_train, X_test, y_train, y_test = prepare_ml_dataset(stock_data)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Model Accuracy: {accuracy}")
    except Exception as e:
        print(f"An error occurred: {e}")
