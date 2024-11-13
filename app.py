import streamlit as st
import pandas as pd
from traderLib import Trader
from logger import initialise_logger

# Initialize the logger for displaying logs in Streamlit
initialise_logger()

# Streamlit app layout
st.title("Trading Bot Analysis")
ticker = st.text_input("Enter Ticker Symbol:")

# Helper function for processing indicators
def process_technical_indicator(value, name):
    """Processes the technical indicator for displaying purposes."""
    if isinstance(value, tuple):
        level, indication = value
    else:
        level = value
        indication = "N/A"
    return level, indication

if ticker:
    trader = Trader(ticker)
    
    try:
        # Fetch historical data for the ticker
        stock_df = trader.get_historical_data(months=18)
        if stock_df is None or stock_df.empty:
            st.warning(f"No data available for {ticker}. Please try a different symbol.")
        else:
            st.write(f"**OHLC Data for {ticker}**", stock_df.tail(5))  # Display the last few rows of data

            # Display the candlestick chart
            st.subheader("Price Chart (Last 12 Months)")
            fig = trader.plot_stock_data(stock_df)
            st.pyplot(fig)

            # Moving Averages
            st.subheader("Moving Averages")
            moving_averages_df = trader.calculate_moving_averages(stock_df)
            st.table(moving_averages_df)

            # Moving Averages Crossovers
            st.subheader("Moving Averages Crossovers")
            moving_average_crossovers_df = trader.calculate_moving_average_crossovers(stock_df)
            st.table(moving_average_crossovers_df)

            # Sentiment Analysis
            st.subheader("Sentiment Analysis")
            sentiment_scores = trader.fetch_and_analyze_news()  # Fetch sentiment scores for each period
            st.write(f"Sentiment Scores: {sentiment_scores}")

            # Technical Indicators
            st.subheader("Technical Indicators")
            technical_indicators = {
                "RSI (14)": trader.calculate_rsi(stock_df),
                "MACD (12,26,9)": trader.calculate_macd(stock_df),
                "Stochastic (20,3)": trader.calculate_stochastic_oscillator(stock_df),
                "ROC (20)": trader.calculate_roc(stock_df),
                "CCI (20)": trader.calculate_cci(stock_df),
                "William %R (14)": trader.calculate_william_percent_r(stock_df),
                "ATR (14)": trader.calculate_atr(stock_df),
                "ADX (14)": trader.calculate_adx(stock_df),
                "Bollinger Bands (20,2)": trader.calculate_bollinger_bands(stock_df),
            }

            # Prepare and display the Technical Indicators table
            indicator_data = {
                "Indicator": [],
                "Level": [],
                "Indication": []
            }
            for name, value in technical_indicators.items():
                level, indication = process_technical_indicator(value, name)
                indicator_data["Indicator"].append(name)
                indicator_data["Level"].append(level)
                indicator_data["Indication"].append(indication)

            st.table(pd.DataFrame(indicator_data))

            # Benchmark for RSC Calculation
            st.subheader("Relative Strength Comparison (RSC)")
            benchmark_ticker = st.text_input("Enter Benchmark Ticker for RSC (Default: SPY)", "SPY")
            benchmark_trader = Trader(benchmark_ticker)
            benchmark_df = benchmark_trader.get_historical_data(months=18)
            if not benchmark_df.empty:
                rsc_value, rsc_indication = trader.calculate_rsc(stock_df, benchmark_df)
                st.write(f"RSC (6 months) for {ticker} vs {benchmark_ticker}: {rsc_value} - {rsc_indication}")
            else:
                st.write("No Benchmark Data for RSC calculation")

            # Final Buy/No Buy Decision
            st.subheader("Buy Decision")
            if trader.should_buy(stock_df):
                st.success(f"**BUY** signal for {ticker}.")
            else:
                st.error(f"**NO BUY** signal for {ticker}.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
