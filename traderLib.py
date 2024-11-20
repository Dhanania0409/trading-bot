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
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

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



def plot_interactive_candlestick(df, ticker):
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=ticker
                )
            ]
        )
        
        # Customize layout
        fig.update_layout(
            title=f"{ticker}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",  # Optional: Choose a theme
            height=600,  # Adjust height
        )
        
        # Add moving averages for better insights (Optional)
        if 'MA_20' in df.columns:  # Assuming MA_20 is calculated
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA_20'],
                mode='lines',
                name='20-Day MA',
                line=dict(color='orange')
            ))
        if 'MA_50' in df.columns:  # Assuming MA_50 is calculated
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MA_50'],
                mode='lines',
                name='50-Day MA',
                line=dict(color='blue')
            ))

        return fig

def plot_sentiment_score(sentiment_scores):
    """Plots sentiment scores for the past 12 months."""
    # Convert sentiment_scores (dictionary) into a DataFrame
    df = pd.DataFrame(list(sentiment_scores.items()), columns=['Period', 'Score'])

    # Create the line chart
    fig = px.line(
        df,
        x='Period',
        y='Score',
        title="Sentiment Analysis Score (12 Months)",
        labels={'Period': 'Period', 'Score': 'Sentiment Score'},
        markers=True
    )
    fig.update_layout(template="plotly_dark")
    return fig

def plot_stochastic_oscillator(df, ticker):
    k_period = 20
    d_period = 3

    high_max = df['High'].rolling(window=k_period).max()
    low_min = df['Low'].rolling(window=k_period).min()
    stoch_k = ((df['Close'] - low_min) * 100) / (high_max - low_min)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=stoch_k, mode='lines', name='%K'))
    fig.add_trace(go.Scatter(x=df.index, y=stoch_d, mode='lines', name='%D'))
    fig.update_layout(
        title=f"{ticker} Stochastic Oscillator",
        xaxis_title="Date",
        yaxis_title="Stochastic Oscillator",
        template="plotly_dark"
    )
    return fig

def plot_adx(df, ticker):

    period = 14

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=adx, mode='lines', name='ADX'))
    fig.update_layout(
        title=f"{ticker} Average Directional Index (ADX)",
        xaxis_title="Date",
        yaxis_title="ADX Value",
        template="plotly_dark"
    )
    return fig


def plot_macd(df, ticker):
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=macd, mode='lines', name='MACD Line'))
    fig.add_trace(go.Scatter(x=df.index, y=signal, mode='lines', name='Signal Line'))
    fig.update_layout(
        title=f"{ticker} MACD",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark"
    )
    return fig

def calculate_daily_cci(df, period=20):
        """
        Calculate the daily Commodity Channel Index (CCI) for plotting.
        
        Args:
            df (pd.DataFrame): The OHLC data.
            period (int): The lookback period for CCI calculation.

        Returns:
            pd.DataFrame: A DataFrame containing daily CCI values.
        """
        # Calculate the typical price
        tp = (df['High'] + df['Low'] + df['Close']) / 3

        # Calculate the simple moving average (SMA) of the typical price
        sma = tp.rolling(window=period).mean()

        # Calculate the mean absolute deviation (MAD)
        mad = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)

        # Calculate the CCI
        cci = (tp - sma) / (0.015 * mad)

        # Return the daily CCI values as a DataFrame
        return pd.DataFrame({'CCI': cci}, index=df.index)

def fetch_historical_macro_data(series_id, start_date, end_date):
    """
    Fetch historical macroeconomic data for a given series ID and date range using the FRED API.
    Args:
        series_id (str): The FRED series ID.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the macroeconomic data.
    """
    FRED_API_KEY = "0594c3e9f8b9be9db5cd44aec538f522"
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&observation_start={start_date}&observation_end={end_date}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract observations and convert to DataFrame
        observations = data.get("observations", [])
        df = pd.DataFrame(observations)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)  # Convert to timezone-naive
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df.set_index("date", inplace=True)
        return df
    except Exception as e:
        lg.error(f"Error fetching macroeconomic data for {series_id}: {e}")
        return pd.DataFrame()


def merge_ohlc_with_macro(ohlc_data, macro_data):
    """
    Merge OHLC data with macroeconomic data.
    Args:
        ohlc_data (pd.DataFrame): Stock OHLC data.
        macro_data (pd.DataFrame): Macroeconomic data.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Ensure both DataFrames have datetime indices
    ohlc_data.index = pd.to_datetime(ohlc_data.index)
    macro_data.index = pd.to_datetime(macro_data.index)

    # Resample macro data to align with OHLC data
    macro_data = macro_data.resample("D").ffill()  # Fill missing values with the last value

    # Merge the data
    merged_data = ohlc_data.merge(macro_data, how="left", left_index=True, right_index=True)
    return merged_data



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
    
    FRED_API_KEY="86948f4a9f0f5afe26d652c945e2970d"
    
    def fetch_macro_data(self):
        """
        Fetch real-time macroeconomic data using the FRED API.
        """
        FRED_API_KEY = "0594c3e9f8b9be9db5cd44aec538f522"  # Replace with your FRED API key
        macro_factors = {}

        # API endpoints for each macroeconomic factor
        endpoints = {
            "10-Year US Treasury Yield": f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={FRED_API_KEY}&file_type=json",
            "Federal Funds Rate": f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={FRED_API_KEY}&file_type=json",
            "Unemployment Rate": f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={FRED_API_KEY}&file_type=json",
            "Consumer Price Index (CPI)": f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={FRED_API_KEY}&file_type=json",
            "GDP Growth Rate": f"https://api.stlouisfed.org/fred/series/observations?series_id=A191RL1Q225SBEA&api_key={FRED_API_KEY}&file_type=json",
        }

        try:
            for factor, url in endpoints.items():
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()

                # Get the latest observation value
                observations = data.get("observations", [])
                if observations:
                    latest_observation = observations[-1]
                    macro_factors[factor] = float(latest_observation["value"])
                else:
                    lg.warning(f"No data available for {factor}. Setting value to 0.")
                    macro_factors[factor] = 0.0

            lg.info(f"Fetched macroeconomic factors: {macro_factors}")
            return macro_factors

        except Exception as e:
            lg.error(f"Error fetching macroeconomic data: {e}")
            return None


    def calculate_macro_score(self):
        """
        Calculate the macroeconomic score based on weighted macro factors.
        """
        macro_data = self.fetch_macro_data()
        if not macro_data:
            lg.error("Failed to fetch macroeconomic data.")
            return 0.0

        # Assign weights to each macroeconomic factor
        weights = {
            "10-Year US Treasury Yield": 0.05,
            "Federal Funds Rate": 0.05,
            "Unemployment Rate": 0.05,
            "Consumer Price Index (CPI)": 0.05,
            "GDP Growth Rate": 0.05,
        }

        # Calculate weighted score
        score = 0.0
        for factor, weight in weights.items():
            if factor in macro_data:
                score += weight * macro_data[factor]

        # Normalize the score (optional)
        normalized_score = score / sum(weights.values())
        lg.info(f"Calculated macroeconomic score: {normalized_score}")
        return normalized_score

    
    def get_historical_data(self, months=12):
        """
        Fetch historical OHLC data using Yahoo Finance, with dynamic column renaming.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months * 30)

            # Fetch data using yfinance
            data = yf.download(
                self.ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )

            # Check if data is empty
            if data.empty:
                lg.error(f"No OHLC data found for {self.ticker} between {start_date} and {end_date}.")
                return None

            # Flatten multi-index columns (if present)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [' '.join(col).strip() for col in data.columns]

            # Dynamically rename columns to standard names
            data.rename(columns=lambda col: col.split()[0] if ' ' in col else col, inplace=True)

            # Convert index to datetime
            data.index = pd.to_datetime(data.index)

            lg.info(f"Fetched {len(data)} rows of OHLC data for {self.ticker} from Yahoo Finance.")
            return data

        except Exception as e:
            lg.error(f"Error fetching OHLC data for {self.ticker}: {e}")
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
        macd_line = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        latest_macd = macd_line.iloc[-1]
        latest_signal = signal_line.iloc[-1]
        indication = "Bullish" if latest_macd > latest_signal else "Bearish"
        return round(latest_macd, 2), round(latest_signal, 2), indication

    def calculate_stochastic_oscillator(self, df, k_period=20, d_period=3):
        high_max = df['High'].rolling(window=k_period).max()
        low_min = df['Low'].rolling(window=k_period).min()
        stoch_k = ((df['Close'] - low_min) * 100) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        latest_k = stoch_k.iloc[-1]
        latest_d = stoch_d.iloc[-1]
        indication = "Overbought" if latest_k > 80 else "Oversold" if latest_k < 20 else "Neutral"
        return round(latest_k, 2), round(latest_d, 2), indication


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
    
    def calculate_fibonacci_retracement(self, df):
        try:
            high_price = df['High'].max()
            low_price = df['Low'].min()
            price_range = high_price - low_price

            fib_levels = {
                '0%': high_price,
                '23.6%': high_price - (price_range * 0.236),
                '38.2%': high_price - (price_range * 0.382),
                '50%': high_price - (price_range * 0.5),
                '61.8%': high_price - (price_range * 0.618),
                '100%': low_price
            }
            lg.info(f"Fibonacci Retracement Levels: {fib_levels}")
            return fib_levels

        except Exception as e:
            lg.error(f"Error calculating Fibonacci retracement levels: {e}")
            return {}


    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        latest_ub = upper_band.iloc[-1]
        latest_lb = lower_band.iloc[-1]
        latest_sma = sma.iloc[-1]
        indication = "Within Bands" if latest_lb <= df['Close'].iloc[-1] <= latest_ub else "Outside Bands"
        return f"UB: {round(latest_ub, 2)}, LB: {round(latest_lb, 2)}, SMA: {round(latest_sma, 2)}, {indication}"
    
    def calculate_sentiment_scores(self):
        try:
            # Example logic to compute sentiment scores
            short_term_score = self.get_sentiment_score(timeframe="1_month")
            medium_term_score = self.get_sentiment_score(timeframe="6_months")
            long_term_score = self.get_sentiment_score(timeframe="12_months")

            return {
                "short_term": short_term_score,
                "medium_term": medium_term_score,
                "long_term": long_term_score
            }
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {e}")
            return {
                "short_term": "N/A",
                "medium_term": "N/A",
                "long_term": "N/A"
            }

    def get_sentiment_score(self, timeframe):
        # Fetch news articles for the specified timeframe
        news_articles = fetch_news_articles(timeframe=timeframe)  # Ensure this function is defined
        if not news_articles:
            logging.warning(f"No news articles found for timeframe {timeframe}.")
            return 0.0
        
        # Sentiment analysis logic using VADER or similar
        sentiment_analyzer = SentimentIntensityAnalyzer()
        cumulative_sentiment_score = 0
        relevant_articles_count = 0
        
        for article in news_articles:
            sentiment = sentiment_analyzer.polarity_scores(article['content'])['compound']
            
            # Filter out neutral sentiment scores
            if abs(sentiment) > 0.05:  # Adjust the threshold if needed
                cumulative_sentiment_score += sentiment
                relevant_articles_count += 1

        # Return the cumulative score or 0.0 if no relevant articles were found
        return cumulative_sentiment_score if relevant_articles_count > 0 else 0.0

    
    def check_moving_average_crossover(self, df):

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
                overall_score += 0.1  # Assign weight for a bullish signal
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

            # Macroeconomic Factors
            macroeconomic_score = self.calculate_macro_score()
            overall_score += macroeconomic_score * 0.25  # Assign 25% total weight
            lg.info(f"Macroeconomic Score Contribution: {macroeconomic_score * 25:.2f}%")

            # Final decision based on accumulated score
            lg.info(f"Overall Score: {overall_score * 100:.2f}%")
            return overall_score >= 0.2  # Set threshold for a buy decision

        except Exception as e:
            lg.error(f"An error occurred in should_buy: {e}")
            return False


