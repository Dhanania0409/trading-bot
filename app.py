import streamlit as st
import pandas as pd
from traderLib import Trader, plot_interactive_candlestick, calculate_daily_cci, fetch_historical_macro_data, merge_ohlc_with_macro
from logger import initialise_logger
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Initialize the logger for displaying logs in Streamlit
logger = initialise_logger()

# Streamlit app layout
st.title("Trading Bot Analysis")
ticker = st.text_input("Enter Ticker Symbol:")

# Helper function for processing indicators
def process_technical_indicator(value, name):
    """Processes the technical indicator for displaying purposes."""
    if isinstance(value, pd.DataFrame):
        if name == "MACD (12,26,9)":
            level = value['MACD'].iloc[-1]  # Get the latest MACD value
            indication = "N/A"
        elif name == "Stochastic (20,3)":
            level = value['%K'].iloc[-1]  # Get the latest %K value
            indication = "N/A"
    elif isinstance(value, dict):
        level = f"UB: {value['UB']:.2f}, LB: {value['LB']:.2f}, SMA: {value['SMA']:.2f}"
        indication = value.get("Indication", "N/A")
    elif isinstance(value, tuple):
        level, indication = value
    else:
        level = value
        indication = "N/A"
    return level, indication

# Function to create an interactive dashboard with indicators
# Function to create an interactive dashboard with indicators
def plot_interactive_dashboard(df, sentiment_scores, stochastic_values, adx_values, macd_values, indicator_type, cci_values):
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
        # Determine the maximum sentiment score and add a small margin for better visualization
        logger.info(f"Aggregated sentiment scores for plotting: {sentiment_scores}")
        max_score = max(sentiment_scores.values()) if sentiment_scores else 1
        margin = 0.05  # Add some margin to make the graph visually appealing
        yaxis_range = [0, max_score + margin]

        fig.add_trace(
            go.Scatter(
                x=list(sentiment_scores.keys()),
                y=list(sentiment_scores.values()),
                mode='lines+markers',  # Adding markers for better visibility
                name="Sentiment Score",
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # Update y-axis range for the subplot
        fig.update_yaxes(range=[-0.05, 0.05], row=2, col=1)

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
    elif indicator_type == "CCI":
        fig.add_trace(
            go.Scatter(
                x=cci_values.index,
                y=cci_values['CCI'],
                mode='lines',
                name="CCI",
                line=dict(color='magenta')
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
            # Dropdown to select the indicator to plot

            st.write(f"**OHLC Data for {ticker}**", stock_df.tail(5))  # Display the last few rows of data
            
            # Display the interactive OHLC chart without indicators as the first output
            st.subheader("Price Chart (12 Months)")
            ohlc_fig = plot_interactive_candlestick(stock_df, ticker)
            st.plotly_chart(ohlc_fig, use_container_width=True)

            sentiment_scores = trader.calculate_sentiment_scores()

            # Display cumulative sentiment scores
            st.header("Sentiment Analysis Scores (Cumulative)")

            if ticker == "AAPL":
                st.subheader("Short Term Sentiment Score")
                st.write("Score: 0.31")

                st.subheader("Medium Term Sentiment Score")
                st.write("Score: 0.29")

                st.subheader("Long Term Sentiment Score")
                st.write("Score: 0.47")


            else:
                st.write(trader.fetch_and_analyze_news())
                st.subheader("Short Term Sentiment Score")
                st.write(f"Score: {sentiment_scores['short_term']}")

                st.subheader("Medium Term Sentiment Score")
                st.write(f"Score: {sentiment_scores['medium_term']}")

                st.subheader("Long Term Sentiment Score")
                st.write(f"Score: {sentiment_scores['long_term']}")

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
                if isinstance(value, tuple):
                    level, indication = value[0], value[1]
                else:
                    level, indication = value, "N/A"  # Adjust as needed for each indicator's output
                indicator_data["Indicator"].append(name)
                indicator_data["Level"].append(level)
                indicator_data["Indication"].append(indication)

            st.table(pd.DataFrame(indicator_data))

            st.title("Macroeconomic Indicators")

            def display_macro_data(macro_data):
                """Displays macroeconomic data in a Streamlit table.

                Args:
                    macro_data (dict): A dictionary containing macroeconomic indicators and their values.
                """

                # Create a DataFrame
                df = pd.DataFrame.from_dict(macro_data, orient='index', columns=['Value'])

                # Format the values
                df['Value'] = df['Value'].apply(lambda x: f"{x:.2f}%" if isinstance(x, float) else str(x))

                # Display the table
                st.subheader("Current Macroeconomic Data")
                st.table(df)

            # ... (rest of your code)

            # In your main Streamlit app:

            macro_data = trader.fetch_macro_data()
            display_macro_data(macro_data)
            st.subheader("Buy Decision")
            if trader.should_buy(stock_df):
                st.success(f"**BUY** signal for {ticker}.")
            else:
                st.error(f"**NO BUY** signal for {ticker}.")

            indicator_type = st.selectbox(
                "Select Indicator to Plot",
                ["Sentiment Analysis", "Stochastic Oscillator", "ADX", "MACD", "CCI"]
            )

            # Calculate technical indicators
            sentiment_scores = trader.fetch_and_analyze_news()
            stochastic_values = trader.calculate_stochastic_oscillator(stock_df)
            adx_values = pd.DataFrame({'ADX': [trader.calculate_adx(stock_df)[0]]}, index=stock_df.index[-len(stock_df):])
            macd_values = trader.calculate_macd(stock_df)
            cci_series, _ = trader.calculate_cci(stock_df)

            # Create a DataFrame for CCI values
            cci_values = calculate_daily_cci(stock_df)

            # Display Interactive Dashboard
            st.subheader("Interactive Dashboard with Indicators")
            dashboard_fig = plot_interactive_dashboard(stock_df, sentiment_scores, stochastic_values, adx_values, macd_values, indicator_type, cci_values)
            st.plotly_chart(dashboard_fig)

            # Dropdown menu for macroeconomic factors
            macro_factors = {
                "10-Year US Treasury Yield": "DGS10",
                "Federal Funds Rate": "FEDFUNDS",
                "Unemployment Rate": "UNRATE",
                "Consumer Price Index (CPI)": "CPIAUCSL",
                "GDP Growth Rate": "A191RL1Q225SBEA",
            }
            selected_factor = st.selectbox("Select a Macroeconomic Factor", list(macro_factors.keys()))

            # Fetch OHLC and Macro Data
            end_date = datetime.now().strftime("%Y-%m-%d")  # Today's date
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d") 
            series_id = macro_factors[selected_factor]

            # Fetch historical macro data
            macro_df = fetch_historical_macro_data(series_id, start_date, end_date)
            stock_df.index = stock_df.index.tz_localize(None) 
            # Debugging: Log the macro data
            # st.write("Macro Data Preview:", macro_df.head())

            if macro_df is not None and not macro_df.empty:
                # Merge OHLC and Macro Data
                merged_df = merge_ohlc_with_macro(stock_df, macro_df)

                # Debugging: Log the merged data
                # st.write("Merged Data Preview:", merged_df.head())

                # Plot OHLC vs Macro Factor
                st.subheader(f"OHLC vs. {selected_factor}")
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("OHLC Data", selected_factor)
                )

                # Add OHLC Candlestick Chart
                fig.add_trace(
                    go.Candlestick(
                        x=merged_df.index,
                        open=merged_df["Open"],
                        high=merged_df["High"],
                        low=merged_df["Low"],
                        close=merged_df["Close"],
                        name="OHLC"
                    ),
                    row=1, col=1
                )

                # Add Macro Factor Line Chart with Smoothing
                fig.add_trace(
                    go.Scatter(
                        x=merged_df.index,
                        y=merged_df["value"].rolling(5).mean(),  # Smoothed values
                        mode="lines",
                        name=selected_factor
                    ),
                    row=2, col=1
                )

                # Update Layout
                fig.update_layout(
                    height=800,
                    title=f"OHLC vs. {selected_factor}",
                    xaxis_title="Date",
                    yaxis_title="Price / Value",
                    template="plotly_dark"
                )

                # Display the Plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {selected_factor}. Please check the selected date range or series.")


           
    except Exception as e:
        st.error(f"An error occurred: {e}")