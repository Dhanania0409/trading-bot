# Import necessary libraries
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Define a variable for the stock ticker
ticker = input("Enter the stock ticker symbol (e.g., AAPL, MSFT): ")

from datetime import datetime, timedelta

# Define the end date as today
end_date = datetime.today().strftime('%Y-%m-%d')

# Define the start date as 10 years ago from today
start_date = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

# Download stock data using the selected ticker
df = yf.download(ticker, start=start_date, end=end_date)

# Add technical indicators
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df = df.dropna()  # Drop NaN values resulting from rolling calculations

# Create a new dataframe with 'Close' and additional technical indicators
data = df[['Close', 'SMA_50', 'SMA_200']]
dataset = data.values  # Convert dataframe to numpy array
training_data_len = math.ceil(len(dataset) * .8)  # 80% of the data for training

# Scale the data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Prepare the training data with a 90-day look-back period
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

look_back_period = 90  # Adjusted to 90 days

for i in range(look_back_period, len(train_data)):
    x_train.append(train_data[i-look_back_period:i, :])  # Include all features
    y_train.append(train_data[i, 0])  # Predicting 'Close' price
    if i <= look_back_period + 1:  # Display the first 2 sequences as an example
        print("x_train:", x_train)
        print("y_train:", y_train)
        print()

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data for LSTM model input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

# Build the GRU model with dropout to reduce overfitting
model = Sequential()
model.add(GRU(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))  # Add dropout to reduce overfitting
model.add(GRU(100, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with early stopping
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(x_train, y_train, batch_size=1, epochs=50, validation_split=0.1, callbacks=[early_stopping])

# Create the test data
test_data = scaled_data[training_data_len - look_back_period:, :]
x_test = []
y_test = dataset[training_data_len:, 0]  # Only the 'Close' column for y_test

for i in range(look_back_period, len(test_data)):
    x_test.append(test_data[i-look_back_period:i, :])

# Convert x_test to a numpy array and reshape for GRU
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# Get the model's predictions for the test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))[:, 0]  # Reverse scaling

# Calculate root mean squared error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print("RMSE:", rmse)

# Plot the training and predicted prices (commented out for now)
# train = data[:training_data_len]
# valid = data[training_data_len:]
# valid['Predictions'] = predictions

# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

# Show the valid and predicted prices
# print(valid)

# Re-download recent stock data using the selected ticker for prediction
apple_quote = yf.download(ticker, start=start_date, end=end_date)

# Ensure 'Close' column exists and extract the last 90 days
if 'Close' in apple_quote.columns:
    new_df = apple_quote[['Close', 'SMA_50', 'SMA_200']]
    new_df = new_df.dropna()  # Drop NaN values due to SMA calculations
    last_90_days = new_df[-look_back_period:].values

    # Check if we have exactly 90 data points
    if len(last_90_days) == look_back_period:
        last_90_days_scaled = scaler.transform(last_90_days)
        
        # Prepare input for prediction
        X_test = []
        X_test.append(last_90_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        
        # Predict and inverse transform to get the actual price
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(np.concatenate((pred_price, np.zeros((pred_price.shape[0], 2))), axis=1))[:, 0]
        print("Predicted Price:", pred_price)
    else:
        print("Error: Insufficient data for the last 90 days.")
else:
    print("Error: 'Close' column not found in the data.")

# Define the end date as today for recent data comparison
end_date = datetime.today().strftime('%Y-%m-%d')

# Define the start date as 5 days before today
start_date = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')

# Download recent stock data for the actual price comparison using the selected ticker
apple_quote2 = yf.download(ticker, start=start_date, end=end_date)
print(apple_quote2['Close'])
