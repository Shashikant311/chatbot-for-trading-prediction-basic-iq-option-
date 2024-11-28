import time
from iqoptionapi.stable_api import IQ_Option
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK setup
nltk.download("punkt")
nltk.download("stopwords")

def connect_to_iqoption(email, password):
    """Connect to IQ Option."""
    print("Connecting to IQ Option...")
    iq = IQ_Option(email, password)
    success, reason = iq.connect()
    if success:
        print("Successfully connected to IQ Option!")
        return iq
    else:
        print(f"Connection failed: {reason}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD and Signal Line."""
    prices_series = pd.Series(prices)
    short_ema = prices_series.ewm(span=short_period, adjust=False).mean()
    long_ema = prices_series.ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def get_prediction_signal(iq, asset):
    """Fetch market data and predict trends based on indicators."""
    print(f"Fetching market data for {asset}...")
    candles = iq.get_candles(asset, 60, 50, time.time())
    if not candles:
        return "Failed to fetch market data."

    prices = np.array([candle['close'] for candle in candles])

    # RSI Calculation
    rsi = calculate_rsi(prices)
    current_rsi = rsi[-1]

    # MACD Calculation
    macd_line, signal_line = calculate_macd(prices)
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]

    # Signal Decision
    if current_rsi > 70:
        return f"{asset}: Overbought condition detected (RSI={current_rsi:.2f}). Possible DOWN trend."
    elif current_rsi < 30:
        return f"{asset}: Oversold condition detected (RSI={current_rsi:.2f}). Possible UP trend."
    elif current_macd > current_signal:
        return f"{asset}: MACD indicates UP trend."
    elif current_macd < current_signal:
        return f"{asset}: MACD indicates DOWN trend."
    else:
        return f"{asset}: Neutral trend detected."

def preprocess_input(user_input):
    """Preprocess user input: tokenize and remove stopwords."""
    tokens = word_tokenize(user_input.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return filtered_tokens

def detect_intent(preprocessed_tokens):
    """Detect the intent based on preprocessed tokens."""
    intents = {
        "ask_timeframe": ["timeframe", "minute", "trade", "1", "5"],
        "ask_trend": ["trend", "up", "down", "signal", "market"],
        "ask_prediction": ["predict", "prediction", "forecast", "signal"],
        "ask_rsi": ["rsi", "relative strength index", "overbought", "oversold"],
        "ask_macd": ["macd", "moving average convergence divergence", "signal line", "macd line"],
        "ask_asset": ["asset", "pair", "currency", "symbol"],
        "ask_best_strategy": ["best strategy", "strategy", "trading method"],
        "ask_suggest_time": ["suggest time", "best time", "execute trade"],
    }

    for intent, keywords in intents.items():
        if any(token in keywords for token in preprocessed_tokens):
            return intent
    return "unknown"

def chatbot():
    """Chatbot interface for market predictions."""
    print("Welcome to Mayuri Jadav's Trading Prediction Chatbot!")
    email = input("Enter your email: ")
    password = input("Enter your password: ")

    iq = connect_to_iqoption(email, password)
    if iq:
        print("\nYou can ask questions about trading predictions. Type 'exit' to quit.")
        while True:
            user_input = input("\nAsk your question: ").strip().lower()

            if user_input == 'exit':
                print("Goodbye! Disconnecting...")
                break

            # Preprocess user input
            tokens = preprocess_input(user_input)
            intent = detect_intent(tokens)

            print(f"Detected Intent: {intent}")

            if intent == "ask_timeframe":
                print(
                    "The best time frame depends on your trading strategy:\n"
                    "- 1-minute trades are highly volatile and suited for scalping.\n"
                    "- 5-minute trades offer slightly more stability for short-term analysis.\n"
                    "Use indicators like RSI and MACD for better timing."
                )
            elif intent == "ask_trend":
                asset = input("Enter the asset (e.g., EURUSD): ").upper()
                signal = get_prediction_signal(iq, asset)
                print(f"Prediction Signal: {signal}")
            elif intent == "ask_prediction":
                asset = input("Enter the asset (e.g., EURUSD): ").upper()
                signal = get_prediction_signal(iq, asset)
                print(f"Prediction Signal: {signal}")
            elif intent == "ask_rsi":
                print("RSI (Relative Strength Index) helps you identify overbought and oversold conditions in the market. It ranges from 0 to 100:\n"
                      "- Over 70: Overbought, potential DOWN trend.\n"
                      "- Below 30: Oversold, potential UP trend.")
            elif intent == "ask_macd":
                print("MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator:\n"
                      "- MACD Line: Difference between short-term and long-term moving averages.\n"
                      "- Signal Line: 9-period EMA of MACD. If MACD crosses above, it's a bullish signal; below, a bearish signal.")
            elif intent == "ask_asset":
                print("You can ask for trading predictions for specific assets (e.g., 'predict EURUSD'). Make sure to enter the correct asset pair or currency symbol.")
            elif intent == "ask_best_strategy":
                print("There is no one-size-fits-all strategy. Some common strategies include:\n"
                      "- Scalping: Short-term trades, often on 1-minute or 5-minute charts.\n"
                      "- Swing trading: Holding trades for several days, focusing on medium-term trends.\n"
                      "- Trend following: Entering trades in the direction of the market trend.")
            elif intent == "ask_suggest_time":
                print("The best time to execute a trade depends on market conditions:\n"
                      "- During high volatility, shorter time frames like 1 minute work best.\n"
                      "- For more stability, 5-minute or higher time frames are recommended.")
            else:
                print(
                    "I can help you with trading predictions and strategies. "
                    "Try asking about:\n"
                    "- Trading time frames (e.g., '1 minute or 5 minute').\n"
                    "- Prediction signals for specific assets (e.g., 'predict EURUSD').\n"
                    "- Whether the trend is 'up or down'.\n"
                    "- RSI and MACD indicators.\n"
                    "- Trading strategies and best times for execution."
                )
    else:
        print("Unable to connect. Please check your credentials.")

if __name__ == "__main__":
    chatbot()
