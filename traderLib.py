import alpaca_trade_api as tradeapi
import logging as lg
import sys
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from sentiment_analysis import fetch_news_articles, analyze_news_sentiment, analyze_sentiment

# Load configuration from config.json
def load_config():
    config_path = r"C:\Users\Asus\Desktop\trading-bot\config.json"  # Ensure the path is correct
    with open(config_path) as config_file:
        return json.load(config_file)

# Load the configuration
config = load_config()

# Initialize Alpaca API (using paper trading environment)
api = tradeapi.REST(config['api_key'], config['api_secret'], base_url='https://paper-api.alpaca.markets')

class Trader:
    def __init__(self, ticker):
        self.ticker = ticker
        lg.info(f'Trader initialized with ticker {ticker}')

    def get_historical_data(self, period='100D'):
        """
        Fetch OHLC data for a larger date range to ensure enough data is retrieved.
        """
        try:
            # Calculate a larger date range, for example, 150 calendar days for 100 trading days
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=150)).strftime('%Y-%m-%d')  # Increased from 100 to 150

            # Fetch historical OHLC data
            bars = api.get_bars(
                self.ticker,
                tradeapi.rest.TimeFrame.Day,
                start=start_date,
                end=end_date,
                limit=100,
                adjustment='raw',
                feed='iex'  # Use IEX data
            )

            # Convert OHLC data to a DataFrame for easy manipulation
            df = pd.DataFrame({
                'Open': [bar.o for bar in bars],
                'High': [bar.h for bar in bars],
                'Low': [bar.l for bar in bars],
                'Close': [bar.c for bar in bars],
                'Volume': [bar.v for bar in bars]
            })

            if df.empty:
                lg.error(f"No OHLC data available for {self.ticker}.")
                sys.exit()

            return df
        except Exception as e:
            lg.error(f'Error fetching OHLC data: {e}')
            sys.exit()


    def calculate_moving_average(self, df, period=50):
        """
        Calculate the moving average for the given period (e.g., 50 days or 100 days).
        """
        if len(df) < period:
            lg.info(f"Not enough data to calculate {period}-day moving average. Using available {len(df)} days.")
            return df['Close'].rolling(window=len(df)).mean().iloc[-1]
        return df['Close'].rolling(window=period).mean().iloc[-1]

    def calculate_rsi(self, df, period=14):
        """
        Calculate the Relative Strength Index (RSI) for the given period (default is 14 days).
        """
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        if avg_loss.iloc[-1] == 0:
            return 100  # If no losses, RSI is 100 (overbought)
        if avg_gain.iloc[-1] == 0:
            return 0    # If no gains, RSI is 0 (oversold)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def check_volume_spike(self, df):
        """
        Check if the current volume is at least 15% higher than the 30-day average volume.
        """
        if len(df) < 30:
            lg.info("Not enough data to calculate 30-day volume average.")
            return False
        
        avg_volume = df['Volume'].rolling(window=30).mean().iloc[-1]
        latest_volume = df['Volume'].iloc[-1]
        return latest_volume >= 1.15 * avg_volume

    def fetch_and_analyze_news(self):
        """
        Fetch the last 5 news articles about the company, display them, and analyze their sentiment.
        """
        lg.info(f'Fetching news articles for {self.ticker}...')
        news_articles = fetch_news_articles(self.ticker)

        if news_articles:
            sentiment_score = 0
            print("\nLatest 5 News Articles and Their Sentiment Scores:")
            for article in news_articles:
                try:
                    article_sentiment = analyze_sentiment(article)
                    sentiment_score += article_sentiment
                    print(f"Article: {article}\nSentiment Score: {article_sentiment}\n")
                except Exception as e:
                    lg.error(f"Error analyzing sentiment for article: {article}. Error: {e}")
                    continue

            lg.info(f"Overall sentiment score for {self.ticker}: {sentiment_score}")
            return sentiment_score
        else:
            lg.info(f"No news articles found for {self.ticker}.")
            return 0

    def should_buy(self, df):
        """
        Decision-making logic with updated conditions:
        1. Adjusted sentiment threshold.
        2. Price comparison with short, mid, and long-term moving averages.
        3. Flexibility with RSI levels.
        4. Lower volume spike threshold.
        """
        last_close = df['Close'].iloc[-1]
        sentiment_score = self.fetch_and_analyze_news()

        # Calculate short, mid, and long-term moving averages
        short_term_ma = self.calculate_moving_average(df, period=20)
        mid_term_ma = self.calculate_moving_average(df, period=50)
        long_term_ma = self.calculate_moving_average(df, period=100)

        rsi = self.calculate_rsi(df)
        volume_spike = self.check_volume_spike(df)

        # Log the indicator values
        lg.info(f"Last Close: {last_close}")
        lg.info(f"Short-Term (20-day) Moving Average: {short_term_ma}")
        lg.info(f"Mid-Term (50-day) Moving Average: {mid_term_ma}")
        lg.info(f"Long-Term (100-day) Moving Average: {long_term_ma}")
        lg.info(f"RSI (14-day): {rsi}")
        lg.info(f"Volume Spike: {volume_spike}")

        # Assign weightages to each indicator (adjusting for flexibility)
        sentiment_weight = 0.5 if sentiment_score >= 3 else 0.0
        price_weight = 0.1 if last_close >= short_term_ma else 0.0
        price_weight += 0.2 if last_close >= mid_term_ma else 0.0
        price_weight += 0.2 if last_close >= long_term_ma else 0.0
        rsi_weight = 0.2 if 50 < rsi < 75 else 0.1 if rsi < 50 else 0.0
        volume_weight = 0.1 if volume_spike else 0.0

        # Calculate the overall score
        overall_score = sentiment_weight + price_weight + rsi_weight + volume_weight

        lg.info(f"Overall Score: {overall_score * 100}%")

        # Define a threshold (e.g., 60%) for issuing a BUY signal
        if overall_score >= 0.6:
            lg.info(f"BUY signal for {self.ticker}. All conditions met.")
            return True
        else:
            lg.info(f"NO BUY signal for {self.ticker}. Conditions not met.")
            return False

    def get_account_info(self):
        """
        Fetch account information using Alpaca API.
        """
        try:
            account = api.get_account()
            lg.info(f'Account balance: {account.cash}')
            return account
        except Exception as e:
            lg.error(f'Error fetching account info: {e}')
            sys.exit()
