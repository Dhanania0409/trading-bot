import alpaca_trade_api as tradeapi
import logging as lg
import sys
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import mplfinance as mpf
import matplotlib.dates as mdates
from sentiment_analysis import fetch_news_articles, analyze_sentiment

logger = lg.getLogger()  
if logger.hasHandlers():
    logger.handlers.clear()

lg.basicConfig(level=lg.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
def load_config():
    config_path = r"C:\Users\Asus\Desktop\trading-bot\config.json"
    with open(config_path) as config_file:
        return json.load(config_file)
config = load_config()

api = tradeapi.REST(config['api_key'], config['api_secret'], base_url='https://paper-api.alpaca.markets')

ALPHA_VANTAGE_API_KEY = '8VSXORD2CC9076Q6'
FRED_API_KEY = '8c910ecaa15798c8f9c8b9a10bfebc96'

class Trader:
    def __init__(self, ticker):
        self.ticker = ticker
        lg.info(f'Trader initialized with ticker {ticker}')

    def get_account_info(self):
        try:
            account = api.get_account() 
            return account
        except Exception as e:
            lg.error(f'Error fetching account info: {e}')
            sys.exit()
        
    def fetch_alpha_vantage_data(self, indicator, symbol, **params):
        base_url = 'https://www.alphavantage.co/query'
        api_params = {
            'function': indicator,
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        api_params.update(params) 
        response = requests.get(base_url, params=api_params)
        try:
            data = response.json()
        except ValueError:
            lg.error("Failed to decode JSON response.")
            return None
        if 'Error Message' in data:
            lg.error(f"Error fetching data for {symbol} using {indicator}: {data.get('Error Message')}")
            return None
        return data

    def get_historical_data(self, months=18):
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=months * 30)).strftime('%Y-%m-%d')
            bars = api.get_bars(
                self.ticker,
                tradeapi.rest.TimeFrame.Day,
                start=start_date,
                end=end_date,
                limit=100,
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
            df['Date'] = pd.to_datetime([bar.t for bar in bars])  
            df.set_index('Date', inplace=True)

            if df.empty:
                lg.error(f"No OHLC data available for {self.ticker}.")
                sys.exit()

            return df
        except Exception as e:
            lg.error(f'Error fetching OHLC data: {e}')
            sys.exit()


    def plot_stock_data(self, df):
        df.index = pd.to_datetime(df.index)
        mpf.plot(
            df,
            type='candle',
            style='charles',
            title=f'{self.ticker} Stock Prices - Last 12 Months',
            ylabel='Price',
            volume=False,
            figsize=(14, 8),
            datetime_format='%Y-%m',
        )
    def calculate_moving_average(self, df, period=50):
        if len(df) < period:
            lg.info(f"Not enough data to calculate {period}-day moving average.")
            return df['Close'].rolling(window=len(df)).mean().iloc[-1] 
        return df['Close'].rolling(window=period).mean().iloc[-1]

    def fetch_and_analyze_news(self):
        lg.info(f'Fetching news articles for {self.ticker}...')

        periods = {
            "1 month": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            "3 months": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
            "6 months": (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        }

        overall_sentiment = 0
        total_articles = 0

        for period_name, period_date in periods.items():
            news_articles = fetch_news_articles(self.ticker, start_date=period_date)
            if news_articles:
                sentiment_score = 0
                article_count = 0
                for article in news_articles:
                    article_sentiment = analyze_sentiment(article)
                    sentiment_score += article_sentiment
                    article_count += 1
                overall_sentiment += sentiment_score
                total_articles += article_count
            else:
                lg.info(f"No news articles found for {self.ticker} in the last {period_name}.")
        if total_articles > 0:
            average_sentiment = overall_sentiment / total_articles
        else:
            average_sentiment = 0

        lg.info(f"Average sentiment score for {self.ticker}: {average_sentiment}")
        return average_sentiment


    def check_moving_average_crossover(self, df):
        short_term_ma = df['Close'].rolling(window=50).mean()
        long_term_ma = df['Close'].rolling(window=200).mean()

        if short_term_ma.iloc[-1] > long_term_ma.iloc[-1] and short_term_ma.iloc[-2] <= long_term_ma.iloc[-2]:
            return "Golden Cross - Buy Signal"
        elif short_term_ma.iloc[-1] < long_term_ma.iloc[-1] and short_term_ma.iloc[-2] >= long_term_ma.iloc[-2]:
            return "Death Cross - Sell Signal"
        return "No Crossover Detected"

    def calculate_macd(self, df):
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            return "MACD Bullish Crossover - Buy Signal"
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            return "MACD Bearish Crossover - Sell Signal"
        return "No MACD Crossover Detected"

    def calculate_bollinger_bands(self, df):
        mid_band = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        upper_band = mid_band + (std * 2)
        lower_band = mid_band - (std * 2)

        return upper_band, lower_band

    def calculate_bollinger_band_width(self, df):
        upper_band, lower_band = self.calculate_bollinger_bands(df)
        band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / lower_band.iloc[-1]
        return band_width

    def calculate_fibonacci_retracement(self, df):
        try:
            high_price = df['High'].max()
            low_price = df['Low'].min()
            price_range = high_price - low_price
            fib_levels = {
                '0%': high_price,
                '23.6%': high_price - (price_range * 0.236),
                '38.2%': high_price - (price_range * 0.382),
                '50%': high_price - (price_range * 0.50),
                '61.8%': high_price - (price_range * 0.618),
                '100%': low_price
            }
            lg.info(f"Calculated Fibonacci Retracement Levels: {fib_levels}")
            return fib_levels

        except Exception as e:
            lg.error(f"Fibonacci retracement calculation failed: {e}")
            return None


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

    def calculate_obv(self):
        obv_data = self.fetch_alpha_vantage_data('OBV', self.ticker,interval='daily')
        if obv_data is None:
            lg.error("Failed to retrieve OBV data.")
            return None
        obv_series = obv_data.get('Technical Analysis: OBV')
        if obv_series is None:
            lg.error("No OBV data found.")
            return None
        obv_values = [float(obv_series[key]['OBV']) for key in obv_series]
        if not obv_values:
            lg.error("OBV values list is empty.")
            return None
        return pd.Series(obv_values)

    def calculate_stochastic_oscillator(self):
        stoch_data = self.fetch_alpha_vantage_data('STOCH', self.ticker, interval='daily', slowkmatype=1, slowdmatype=1)
        if stoch_data is None:
            lg.error("Failed to retrieve STOCH data.")
            return None

        stoch_series = stoch_data.get('Technical Analysis: STOCH')
        if stoch_series is None:
            lg.error("No STOCH data found.")
            return None

        stoch_values = [(float(stoch_series[key]['SlowK']) + float(stoch_series[key]['SlowD'])) / 2 for key in stoch_series]
        return pd.Series(stoch_values)

    def check_volume_spike(self, df):
        if len(df) < 30:
            lg.info("Not enough data to calculate 30-day volume average.")
            return False

        avg_volume = df['Volume'].rolling(window=30).mean().iloc[-1]
        latest_volume = df['Volume'].iloc[-1]
        return latest_volume >= 1.15 * avg_volume

    def get_macroeconomic_factors(self):
        macroeconomic_indicators = [
            'CPIAUCSL', 'UNRATE', 'GDP', 'FEDFUNDS', 'M2', 'PPIACO', 'DEXUSEU', 'DGS10', 'PAYEMS'
        ]

        base_url = 'https://api.stlouisfed.org/fred/series/observations'
        headers = {'Content-Type': 'application/json'}

        macro_factors = []
        missing_factors = []
        for indicator in macroeconomic_indicators:
            params = {
                'series_id': indicator,
                'api_key': FRED_API_KEY,
                'file_type': 'json'
            }

            try:
                lg.info(f"Fetching data for {indicator}...")
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                if data.get('observations'):
                    latest_value = float(data['observations'][-1]['value'])
                    macro_factors.append(latest_value)
                    lg.info(f"Fetched {indicator}: {latest_value}")
                else:
                    lg.warning(f"No data found for {indicator}. Adding to missing_factors.")
                    missing_factors.append(indicator)

            except requests.exceptions.RequestException as e:
                lg.error(f"Error fetching macroeconomic data for {indicator}: {e}")
                missing_factors.append(indicator)

        if macro_factors:
            average_macro_factor = sum(macro_factors) / len(macro_factors)
            lg.info(f"Average Macroeconomic Factor: {average_macro_factor}")
            if missing_factors:
                lg.warning(f"Missing data for the following indicators: {missing_factors}")
            return average_macro_factor
        else:
            lg.error("All macroeconomic data requests failed. Returning -1.")
            return -1 

    def should_buy(self, df):
        if df is None or df.empty:
            lg.error("No valid OHLC data to analyze.")
            return False

        try:
            last_close = df['Close'].iloc[-1]
            lg.info("Analyzing sentiment score...")
            sentiment_score = self.fetch_and_analyze_news()
            if sentiment_score is None:
                lg.error("Failed to analyze sentiment.")
                return False
            lg.info("Checking Moving Average Crossover...")
            ma_crossover = self.check_moving_average_crossover(df)
            if ma_crossover is None:
                lg.error("Moving Average Crossover failed.")
                return False
            lg.info(f"Moving Average Crossover result: {ma_crossover}")

            lg.info("Calculating MACD...")
            macd_signal = self.calculate_macd(df)
            if macd_signal is None:
                lg.error("MACD calculation failed.")
                return False
            lg.info(f"MACD result: {macd_signal}")

            lg.info("Calculating Bollinger Bands...")
            upper_band, lower_band = self.calculate_bollinger_bands(df)
            if upper_band is None or lower_band is None:
                lg.error("Bollinger Bands calculation failed.")
                return False
            lg.info(f"Bollinger Bands: Upper {upper_band.iloc[-1]}, Lower {lower_band.iloc[-1]}")

            if last_close > upper_band.iloc[-1]:
                lg.info("Bollinger Bands Breakout (Above Upper Band) - Potential Buy Signal")
            elif last_close < lower_band.iloc[-1]:
                lg.info("Bollinger Bands Breakout (Below Lower Band) - Potential Sell Signal")
            lg.info("Calculating ADX...")
            adx_value = self.calculate_adx(df)
            if adx_value is None:
                lg.error("ADX calculation failed.")
                return False
            lg.info(f"ADX Value: {adx_value} - Trend Strength")
            lg.info("Checking for Volume Spike...")
            volume_spike = self.check_volume_spike(df)
            if volume_spike is None:
                lg.error("Volume spike calculation failed.")
                return False
            lg.info(f"Volume Spike: {volume_spike}")
            lg.info("Calculating Fibonacci Retracement...")
            fibonacci_levels = self.calculate_fibonacci_retracement(df)
            if fibonacci_levels is None:
                lg.error("Fibonacci retracement calculation failed.")
                return False
            lg.info(f"Fibonacci Retracement Levels: {fibonacci_levels}")
            lg.info("Calculating OBV...")
            obv_values = self.calculate_obv()
            if obv_values is None:
                lg.error("OBV calculation failed.")
            else:
                lg.info(f"OBV: {obv_values.iloc[-1]}")
            lg.info("Calculating Stochastic Oscillator...")
            stochastic_oscillator = self.calculate_stochastic_oscillator()
            if stochastic_oscillator is None:
                lg.error("Stochastic Oscillator calculation failed.")
            else:
                lg.info(f"Stochastic Oscillator: {stochastic_oscillator.iloc[-1]}")
            lg.info("Calculating Macroeconomic Factor Average...")
            macro_factor = self.get_macroeconomic_factors()
            if macro_factor is None:
                lg.error("Macroeconomic factor calculation failed.")
                return False
            lg.info(f"Macroeconomic Factor Average: {macro_factor}")
            sentiment_weight = 0.1 if sentiment_score >= 3 else 0.0
            ma_weight = 0.1 if "Golden Cross" in ma_crossover else 0.0
            macd_weight = 0.1 if "Bullish Crossover" in macd_signal else 0.0
            adx_weight = 0.1 if adx_value >= 25 else 0.0
            volume_weight = 0.05 if volume_spike else 0.0
            fibonacci_weight = 0.1 if last_close > fibonacci_levels[-1] else 0.0
            obv_weight = 0.05 if obv_values.iloc[-1] > obv_values.mean() else 0.0
            stochastic_weight = 0.1 if stochastic_oscillator.iloc[-1] > 20 else 0.0
            macro_weight = 0.2 if macro_factor >= 50 else 0.0

            lg.info(f"sentiment_weight: {sentiment_weight}, ma_weight: {ma_weight}, macd_weight: {macd_weight}, adx_weight: {adx_weight}, volume_weight: {volume_weight}, fibonacci_weight: {fibonacci_weight}, obv_weight: {obv_weight}, stochastic_weight: {stochastic_weight}, macro_weight: {macro_weight}")

            overall_score = (sentiment_weight + ma_weight + macd_weight + adx_weight +
                            volume_weight + fibonacci_weight + obv_weight + stochastic_weight +
                            macro_weight)

            lg.info(f"Overall Score: {overall_score * 100}%")
            if overall_score >= 0.6:
                lg.info(f"BUY signal for {self.ticker}. All conditions met.")
                print(f"BUY signal for {self.ticker}.")
                return True
            else:
                lg.info(f"NO BUY signal for {self.ticker}. Conditions not met.")
                print(f"NO BUY signal for {self.ticker}.")
                return False

        except Exception as e:
            lg.error(f"An error occurred: {e}")
            return False

