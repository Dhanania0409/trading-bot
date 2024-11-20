import streamlit as st
import pandas as pd
from traderLib import Trader, plot_interactive_candlestick, plot_sentiment_score
from logger import initialise_logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize the logger for displaying logs in Streamlit
initialise_logger()

# Streamlit app layout
st.title("Trading Bot Analysis")
ticker = st.text_input("Enter Ticker Symbol:")

# Helper function for processing indicators
def process_technical_indicator(value, name):
    """Processes the technical indicator for displaying purposes."""
    if isinstance(value, tuple):
        # If it's a tuple, unpack it
        level, indication = value
    else:
        # If it's not a tuple, return a default value for easier debugging
        level = value
        indication = "N/A"
    return level, indication

# Function to create an interactive dashboard with indicators
def plot_interactive_dashboard(df, sentiment_scores, stochastic_values, adx_values, macd_values, indicator_type):
    # Create subplots with shared x-axes for OHLC chart and one indicator
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.3,  # Increase the vertical spacing between subplots
        subplot_titles=("OHLC Chart", indicator_type if indicator_type else "")
    )

    # OHLC chart with range slider
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )

    # Plot the selected indicator only in the second row
    if indicator_type == "Sentiment Analysis":
        fig.add_trace(
            go.Scatter(
                x=list(sentiment_scores.keys()),
                y=list(sentiment_scores.values()),
                mode='lines',
                name="Sentiment Score",
                line=dict(color='blue')
            ),
            row=2, col=1
        )
    elif indicator_type == "Stochastic Oscillator":
        fig.add_trace(
            go.Scatter(
                x=stochastic_values.index,
                y=stochastic_values['%K'],
                mode='lines',
                name="%K",
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=stochastic_values.index,
                y=stochastic_values['%D'],
                mode='lines',
                name="%D",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
    elif indicator_type == "ADX":
        fig.add_trace(
            go.Scatter(
                x=adx_values.index,
                y=adx_values['ADX'],
                mode='lines',
                name="ADX",
                line=dict(color='green')
            ),
            row=2, col=1
        )
    elif indicator_type == "MACD":
        fig.add_trace(
            go.Scatter(
                x=macd_values.index,
                y=macd_values['MACD'],
                mode='lines',
                name="MACD",
                line=dict(color='red')
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=macd_values.index,
                y=macd_values['Signal'],
                mode='lines',
                name="Signal Line",
                line=dict(color='blue')
            ),
            row=2, col=1
        )

    # Update layout to adjust height and add range slider
    fig.update_layout(
        height=800,
        title_text="Interactive Trading Dashboard",
        showlegend=True,
        xaxis_rangeslider_visible=True
    )
    fig.update_xaxes(title_text="Date")

    return fig


if ticker:
    trader = Trader(ticker)
    
    try:
        # Fetch historical data for the ticker
        stock_df = trader.get_historical_data(months=18)
        if stock_df is None or stock_df.empty:
            st.warning(f"No data available for {ticker}. Please try a different symbol.")
        else:
            # Display the interactive OHLC chart without indicators as the first output
            st.subheader("Price Chart (12 Months)")
            ohlc_fig = plot_interactive_candlestick(stock_df, ticker)
            st.plotly_chart(ohlc_fig, use_container_width=True)

            # Display technical analysis tables and indicators as usual
            st.write(f"**OHLC Data for {ticker}**", stock_df.tail(5))  # Display the last few rows of data

            # Moving Averages
            st.subheader("Moving Averages")
            moving_averages_df = trader.calculate_moving_averages(stock_df)
            st.table(moving_averages_df)

            # Moving Averages Crossovers
            st.subheader("Moving Averages Crossovers")
            moving_average_crossovers_df = trader.calculate_moving_average_crossovers(stock_df)
            st.table(moving_average_crossovers_df)

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

            # Final Buy/No Buy Decision
            st.subheader("Buy Decision")
            if trader.should_buy(stock_df):
                st.success(f"**BUY** signal for {ticker}.")
            else:
                st.error(f"**NO BUY** signal for {ticker}.")

            # Dropdown to select the indicator to plot
            indicator_type = st.selectbox(
                "Select Indicator to Plot",
                ["Sentiment Analysis", "Stochastic Oscillator", "ADX", "MACD"]
            )

            # Calculate technical indicators
            sentiment_scores = trader.fetch_and_analyze_news()
            stochastic_values = trader.calculate_stochastic_oscillator(stock_df)
            adx_values = pd.DataFrame({'ADX': [trader.calculate_adx(stock_df)[0]]}, index=stock_df.index[-len(stock_df):])
            macd_values = trader.calculate_macd(stock_df)

            # Display Interactive Dashboard with the selected indicator
            st.subheader("Interactive Dashboard with Indicators")
            dashboard_fig = plot_interactive_dashboard(stock_df, sentiment_scores, stochastic_values, adx_values, macd_values, indicator_type)
            st.plotly_chart(dashboard_fig)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
