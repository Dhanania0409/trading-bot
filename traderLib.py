import alpaca_trade_api as tradeapi
import logging as lg
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import mplfinance as mpf
import numpy as np
from pytz import timezone
from sentiment_analysis import fetch_news_articles, analyze_news_sentiment

# Logger setup
logger = lg.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
lg.basicConfig(level=lg.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load API configuration
def load_config():
    config_path = r"C:\Users\Asus\Desktop\trading-bot\config.json"
    with open(config_path) as config_file:
        return json.load(config_file)

config = load_config()
api = tradeapi.REST(config['api_key'], config['api_secret'], base_url='https://paper-api.alpaca.markets')

class Trader:
    def __init__(self, ticker):
        self.ticker = ticker
        lg.info(f'Trader initialized with ticker {ticker}')

    def fetch_and_analyze_news(self):
        lg.info(f'Fetching news articles for {self.ticker}...')
        
        periods = {
            "1 month": datetime.now() - timedelta(days=30),
            "6 months": datetime.now() - timedelta(days=180),
            "12 months": datetime.now() - timedelta(days=365)
        }

        sentiment_scores = {}

        for period_name, start_date in periods.items():
            news_articles = fetch_news_articles(self.ticker, start_date=start_date.strftime('%Y-%m-%d'))

            if news_articles:
                sentiment_score = analyze_news_sentiment(news_articles)
                sentiment_scores[period_name] = round(sentiment_score / len(news_articles), 2)
            else:
                lg.info(f"No news articles found for {self.ticker} in the last {period_name}.")
                sentiment_scores[period_name] = 0

        return sentiment_scores

    def get_historical_data(self, months=12):
        try:
            tz = timezone('US/Eastern')
            end_date = tz.localize(datetime.now()).strftime('%Y-%m-%d')
            start_date = tz.localize(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            bars = api.get_bars(
                self.ticker,
                tradeapi.rest.TimeFrame.Day,
                start=start_date,
                end=end_date,
                limit=500,
                adjustment='raw',
                feed='iex'
            )
            df = pd.DataFrame({
                'Open': [bar.o for bar in bars],
                'High': [bar.h for bar in bars],
                'Low': [bar.l for bar in bars],
                'Close': [bar.c for bar in bars],
                'Volume': [bar.v for bar in bars]
            })
            df['Date'] = pd.to_datetime([bar.t for bar in bars]).tz_localize(None)  
            df.set_index('Date', inplace=True)

            if df.empty or df.index[0] > pd.to_datetime(start_date):
                lg.error(f"Insufficient OHLC data for a full 12 months for {self.ticker}.")
                return None

            return df
        except Exception as e:
            lg.error(f'Error fetching OHLC data for {self.ticker}: {e}')
            return None

    def plot_stock_data(self, df):
        df.index = pd.to_datetime(df.index)
        fig, ax = plt.subplots(figsize=(14, 8))
        mpf.plot(
            df,
            type='candle',
            style='charles',
            ax=ax,
            volume=False,
            datetime_format='%Y-%m',
            show_nontrading=True
        )
        ax.set_title(f'{self.ticker} Stock Prices - Last 12 Months')
        ax.set_ylabel('Price')
        return fig

    # Technical Indicator Calculations
    def calculate_moving_averages(self, df):
        periods = [5, 10, 20, 50, 100, 200]
        ma_values = []
        
        for period in periods:
            if len(df) >= period:
                ma_value = df['Close'].rolling(window=period).mean().iloc[-1]
            else:
                ma_value = df['Close'].mean()
            indication = "Bullish" if df['Close'].iloc[-1] > ma_value else "Bearish"
            ma_values.append({"Period": period, "SMA": round(ma_value, 2), "Indication": indication})
        
        return pd.DataFrame(ma_values)

    def calculate_rsi(self, df, period=14):
        if len(df) < period:
            lg.warning("Insufficient data for RSI calculation.")
            return None, "Not enough data"
        
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi.iloc[-1]
        indication = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
        return latest_rsi, indication

    def calculate_macd(self, df):
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = macd.iloc[-1]
        macd_signal = signal.iloc[-1]
        indication = "Bullish" if macd_value > macd_signal else "Bearish"
        return macd_value - macd_signal, indication

    def calculate_stochastic_oscillator(self, df, k_period=20, d_period=3):
        high_max = df['High'].rolling(window=k_period).max()
        low_min = df['Low'].rolling(window=k_period).min()
        stoch_k = ((df['Close'] - low_min) * 100) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        latest_stoch = stoch_d.iloc[-1]
        indication = "Overbought" if latest_stoch > 80 else "Oversold" if latest_stoch < 20 else "Neutral"
        return latest_stoch, indication

    def calculate_roc(self, df, period=20):
        roc = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        latest_roc = roc.iloc[-1]
        indication = "Bullish" if latest_roc > 0 else "Bearish"
        return latest_roc, indication

    def calculate_cci(self, df, period=20):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        latest_cci = cci.iloc[-1]
        indication = "Bullish" if latest_cci > 100 else "Bearish" if latest_cci < -100 else "Neutral"
        return latest_cci, indication

    def calculate_william_percent_r(self, df, period=14):
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        will_r = ((high_max - df['Close']) / (high_max - low_min)) * -100
        latest_will_r = will_r.iloc[-1]
        indication = "Overbought" if latest_will_r < -80 else "Oversold" if latest_will_r > -20 else "Neutral"
        return latest_will_r, indication

    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, high_close, low_close)
        atr = true_range.rolling(window=period).mean()
        latest_atr = atr.iloc[-1]
        indication = "Low Volatility" if latest_atr < 1 else "High Volatility"
        return latest_atr, indication
    
    def calculate_moving_average_crossovers(self, df):
        crossovers = []
        
        crossover_periods = [
            (5, 20, "Short Term"),
            (20, 50, "Medium Term"),
            (50, 200, "Long Term")
        ]
        
        for short_period, long_period, term in crossover_periods:
            if len(df) < long_period:
                indication = "No Data"  
                crossover_type = f"{short_period} & {long_period} DMA Crossover"
            else:
                short_ma = df['Close'].rolling(window=short_period).mean().iloc[-1]
                long_ma = df['Close'].rolling(window=long_period).mean().iloc[-1]
                indication = "Bullish" if short_ma > long_ma else "Bearish"
                crossover_type = f"{short_period} & {long_period} DMA Crossover"
            
            crossovers.append({
                "Period": term,
                "Moving Average Crossover": crossover_type,
                "Indication": indication
            })
        
        return pd.DataFrame(crossovers)
    
    def calculate_rsc(self, df, benchmark_df):
        """
        Calculates the Relative Strength Comparison (RSC) with a benchmark.
        """
        if len(df) < 126 or len(benchmark_df) < 126:  # Check for 6 months of data (~126 trading days)
            return None, "Not enough data"

        stock_return = (df['Close'].iloc[-1] - df['Close'].iloc[-126]) / df['Close'].iloc[-126] * 100
        benchmark_return = (benchmark_df['Close'].iloc[-1] - benchmark_df['Close'].iloc[-126]) / benchmark_df['Close'].iloc[-126] * 100
        rsc = stock_return / benchmark_return
        indication = "Outperformer" if rsc > 1 else "Underperformer"
        
        return round(rsc, 2), indication

    def calculate_adx(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        if adx.empty or pd.isna(adx.iloc[-1]):
            lg.error("ADX calculation resulted in no data or NaN.")
            return None, "No Data"
        
        latest_adx = adx.iloc[-1]
        indication = "Strong Trend" if latest_adx >= 25 else "Weak Trend"
        return round(latest_adx, 2), indication
    
    def check_volume_spike(self, df, window=30, spike_multiplier=1.5):
        """
        Checks if there is a significant volume spike in the recent trading data.

        Parameters:
        - df: The DataFrame containing historical OHLCV data.
        - window: The period over which to calculate the average volume (default is 30 days).
        - spike_multiplier: The factor by which recent volume must exceed the average to be considered a spike.

        Returns:
        - bool: True if there is a volume spike, False otherwise.
        """
        if len(df) <window:
            lg.warning("Not enough data to calculate the volume spike.")
            return False
        
        # Calculate the average volume over the specified window
        avg_volume = df['Volume'].rolling(window=window).mean().iloc[-1]
        # Get the most recent day's volume
        latest_volume = df['Volume'].iloc[-1]

        # Determine if the recent volume exceeds the threshold for a spike
        is_spike = latest_volume > spike_multiplier * avg_volume
        lg.info(f"Volume Spike Check - Latest Volume: {latest_volume}, Average Volume: {avg_volume}, Spike Detected: {is_spike}")

        return is_spike


    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        latest_ub = upper_band.iloc[-1]
        latest_lb = lower_band.iloc[-1]
        latest_sma = sma.iloc[-1]
        indication = "Above Upper Band" if df['Close'].iloc[-1] > latest_ub else "Below Lower Band" if df['Close'].iloc[-1] < latest_lb else "Within Bands"
        return {"UB": round(latest_ub, 2), "LB": round(latest_lb, 2), "SMA": round(latest_sma, 2), "Indication": indication}
    
    def check_moving_average_crossover(self, df):
        """
        Checks for moving average crossovers: Golden Cross or Death Cross.
        """
        short_term_ma = df['Close'].rolling(window=50).mean()
        long_term_ma = df['Close'].rolling(window=200).mean()

        # Check for a Golden Cross (bullish signal)
        if short_term_ma.iloc[-1] > long_term_ma.iloc[-1] and short_term_ma.iloc[-2] <= long_term_ma.iloc[-2]:
            return "Golden Cross - Buy Signal"
        
        # Check for a Death Cross (bearish signal)
        elif short_term_ma.iloc[-1] < long_term_ma.iloc[-1] and short_term_ma.iloc[-2] >= long_term_ma.iloc[-2]:
            return "Death Cross - Sell Signal"
        
        # No crossover detected
        return "No Crossover Detected"

    def should_buy(self, df):
        """
        Determines whether to buy based on multiple indicators, including technical analysis,
        sentiment analysis, and volume spikes.

        Parameters:
        - df: The DataFrame containing historical OHLCV data.

        Returns:
        - bool: True if buy conditions are met, False otherwise.
        """
        if df is None or df.empty:
            lg.error("No valid OHLC data to analyze.")
            return False

        try:
            # Initialize a score that will accumulate weights from different indicators
            overall_score = 0

            # Last close price
            last_close = df['Close'].iloc[-1]

            # Analyze moving average crossover
            ma_crossover = self.check_moving_average_crossover(df)
            if ma_crossover == "Golden Cross - Buy Signal":
                overall_score += 0.15  # Assign weight for a bullish signal
                lg.info("Moving Average Crossover - Bullish (Golden Cross)")
            elif ma_crossover == "Death Cross - Sell Signal":
                lg.info("Moving Average Crossover - Bearish (Death Cross)")

            # RSI Indicator
            rsi_value, rsi_indication = self.calculate_rsi(df)
            if rsi_indication == "Oversold":
                overall_score += 0.1  # Assign weight if RSI is oversold (bullish signal)
                lg.info("RSI - Bullish (Oversold)")

            # MACD Indicator
            macd_value, macd_indication = self.calculate_macd(df)
            if macd_indication == "Bullish":
                overall_score += 0.1  # Assign weight if MACD is bullish
                lg.info("MACD - Bullish Crossover")

            # Bollinger Bands Indicator
            bollinger_bands = self.calculate_bollinger_bands(df)
            if bollinger_bands and last_close < bollinger_bands["LB"]:
                overall_score += 0.1  # Assign weight if close price is below lower Bollinger Band
                lg.info("Bollinger Bands - Bullish (Below Lower Band)")

            # ADX Indicator
            adx_value, adx_indication = self.calculate_adx(df)
            if adx_indication == "Strong Trend" and adx_value >= 25:
                overall_score += 0.1  # Assign weight if ADX indicates a strong trend
                lg.info("ADX - Strong Trend")

            # Stochastic Oscillator
            stochastic_value, stochastic_indication = self.calculate_stochastic_oscillator(df)
            if stochastic_indication == "Oversold":
                overall_score += 0.1  # Assign weight if Stochastic Oscillator indicates oversold
                lg.info("Stochastic Oscillator - Bullish (Oversold)")

            # Volume Spike
            if self.check_volume_spike(df):
                overall_score += 0.05  # Assign weight for a volume spike (potential bullish signal)
                lg.info("Volume Spike - Potential Buy Signal")

            # Fibonacci Retracement
            fibonacci_levels = self.calculate_fibonacci_retracement(df)
            if last_close > fibonacci_levels.get('38.2%', 0):
                overall_score += 0.1  # Assign weight if close price is above the 38.2% retracement level
                lg.info("Fibonacci Retracement - Above 38.2% level")

            # Sentiment Analysis
            sentiment_scores = self.fetch_and_analyze_news()
            average_sentiment_score = sum(sentiment_scores.values()) / len(sentiment_scores) if sentiment_scores else 0
            if average_sentiment_score > 0.2:  # Adjust threshold as necessary
                overall_score += 0.15  # Assign weight based on sentiment score
                lg.info(f"Sentiment Analysis - Bullish (Score: {average_sentiment_score})")

            # Final decision based on accumulated score
            lg.info(f"Overall Score: {overall_score * 100}%")
            return overall_score >= 0.6  # Set threshold for a buy decision

        except Exception as e:
            lg.error(f"An error occurred in should_buy: {e}")
            return False

