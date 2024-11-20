import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from traderLib import fetch_historical_macro_data, merge_ohlc_with_macro  # Ensure these are imported
from datetime import datetime, timedelta
# Mock OHLC Data for Testing
def generate_mock_ohlc_data():
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")  # Business days
    ohlc_data = pd.DataFrame({
        "Date": date_range,
        "Open": pd.Series(range(len(date_range))) + 100,
        "High": pd.Series(range(len(date_range))) + 105,
        "Low": pd.Series(range(len(date_range))) + 95,
        "Close": pd.Series(range(len(date_range))) + 100,
        "Volume": pd.Series(range(len(date_range))) * 1000
    }).set_index("Date")
    return ohlc_data

# Initialize the Streamlit app
st.title("Demo: OHLC vs. Macroeconomic Factors")

# Fetch Mock OHLC Data
stock_df = generate_mock_ohlc_data()
st.write("OHLC Data Preview:", stock_df.head())

# Define Macroeconomic Factors
macro_factors = {
    "10-Year US Treasury Yield": "DGS10",
    "Federal Funds Rate": "FEDFUNDS",
    "Unemployment Rate": "UNRATE",
    "Consumer Price Index (CPI)": "CPIAUCSL",
    "GDP Growth Rate": "A191RL1Q225SBEA",
}

# Dropdown for Macroeconomic Factor Selection
selected_factor = st.selectbox("Select a Macroeconomic Factor", list(macro_factors.keys()))

# Dynamically define start_date and end_date for the last 12 months
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

st.write(f"Fetching data for {selected_factor} from {start_date} to {end_date}...")

# Fetch historical macro data
series_id = macro_factors[selected_factor]
macro_df = fetch_historical_macro_data(series_id, start_date, end_date)

# Debug: Display fetched macro data
st.write("Macro Data Preview:", macro_df)

if macro_df is not None and not macro_df.empty:
    # Merge OHLC and Macro Data
    merged_df = merge_ohlc_with_macro(stock_df, macro_df)

    # Debug: Display merged data
    st.write("Merged Data Preview:", merged_df.head())

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
