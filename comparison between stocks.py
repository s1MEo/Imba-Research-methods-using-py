import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Function to plot stock predictions for multiple stocks
def plot_stock_predictions_multi(models, scalers, stock_datas, x_tests, start_date, end_date, prediction_days, stock_symbols):
    plt.figure(figsize=(12, 6))

    # Iterate over each stock
    for i, (model, scaler, stock_data, x_test, stock_symbol) in enumerate(zip(models, scalers, stock_datas, x_tests, stock_symbols)):
        # Get actual prices
        actual_prices = stock_data['Close'].values

        # Get predicted prices
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))  # Reshape for inverse_transform
        model_inputs = np.concatenate((np.full((prediction_days, 1), np.nan), predicted_prices[:-prediction_days]))

        # Plot actual prices
        plt.plot(stock_data.index[-len(actual_prices):], actual_prices, label=f"Actual {stock_symbol} Price (Test Period) {i+1}")

        # Plot predicted prices
        predicted_prices_with_sign = np.full((len(model_inputs), 1), np.nan)
        predicted_prices_with_sign[-len(predicted_prices):] = predicted_prices
        plt.plot(stock_data.index[-len(model_inputs):], predicted_prices_with_sign, label=f"Predicted {stock_symbol} Price (Test Period) {i+1}")

    # Plot vertical lines for test start and end
    plt.axvline(start_date, color='red', linestyle='--', linewidth=2, label='Test Start')
    plt.axvline(end_date, color='red', linestyle='--', linewidth=2, label='Test End')

    plt.title("Stock Price Predictions")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Load data for S&P 500 and FTSE 100
sp500_symbol = '^GSPC'  # S&P 500
ftse100_symbol = '^FTSE'  # FTSE 100
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime.now()

sp500_data = yf.download(sp500_symbol, start=start_date, end=end_date)
ftse100_data = yf.download(ftse100_symbol, start=start_date, end=end_date)

sp500_scaler = MinMaxScaler(feature_range=(0, 1))
ftse100_scaler = MinMaxScaler(feature_range=(0, 1))

sp500_scaled_data = sp500_scaler.fit_transform(sp500_data['Close'].values.reshape(-1, 1))
ftse100_scaled_data = ftse100_scaler.fit_transform(ftse100_data['Close'].values.reshape(-1, 1))

prediction_days = 60

# Prepare training data for S&P 500
sp500_x_train, sp500_y_train = [], []

for x in range(prediction_days, len(sp500_scaled_data)):
    sp500_x_train.append(sp500_scaled_data[x - prediction_days:x, 0])
    sp500_y_train.append(sp500_scaled_data[x, 0])

sp500_x_train, sp500_y_train = np.array(sp500_x_train), np.array(sp500_y_train)
sp500_x_train = np.reshape(sp500_x_train, (sp500_x_train.shape[0], sp500_x_train.shape[1], 1))

# Build the Model for S&P 500
sp500_model = Sequential()
sp500_model.add(LSTM(units=50, return_sequences=True, input_shape=(sp500_x_train.shape[1], 1)))
sp500_model.add(Dropout(0.2))
sp500_model.add(LSTM(units=50, return_sequences=True))
sp500_model.add(Dropout(0.2))
sp500_model.add(LSTM(units=50))
sp500_model.add(Dropout(0.2))
sp500_model.add(Dense(units=1))
sp500_model.compile(optimizer='adam', loss='mean_squared_error')

# Introduce EarlyStopping callback
early_stopping_sp500 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
sp500_model.fit(sp500_x_train, sp500_y_train, epochs=25, batch_size=64, validation_split=0.1, callbacks=[early_stopping_sp500])

# Prepare test data for S&P 500
sp500_x_test = []

for x in range(prediction_days, len(sp500_scaled_data)):
    sp500_x_test.append(sp500_scaled_data[x - prediction_days:x, 0])

sp500_x_test = np.array(sp500_x_test)
sp500_x_test = np.reshape(sp500_x_test, (sp500_x_test.shape[0], sp500_x_test.shape[1], 1))

# Prepare training data for FTSE 100
ftse100_x_train, ftse100_y_train = [], []

for x in range(prediction_days, len(ftse100_scaled_data)):
    ftse100_x_train.append(ftse100_scaled_data[x - prediction_days:x, 0])
    ftse100_y_train.append(ftse100_scaled_data[x, 0])

ftse100_x_train, ftse100_y_train = np.array(ftse100_x_train), np.array(ftse100_y_train)
ftse100_x_train = np.reshape(ftse100_x_train, (ftse100_x_train.shape[0], ftse100_x_train.shape[1], 1))

# Build the Model for FTSE 100
ftse100_model = Sequential()
ftse100_model.add(LSTM(units=50, return_sequences=True, input_shape=(ftse100_x_train.shape[1], 1)))
ftse100_model.add(Dropout(0.2))
ftse100_model.add(LSTM(units=50, return_sequences=True))
ftse100_model.add(Dropout(0.2))
ftse100_model.add(LSTM(units=50))
ftse100_model.add(Dropout(0.2))
ftse100_model.add(Dense(units=1))
ftse100_model.compile(optimizer='adam', loss='mean_squared_error')

# Introduce EarlyStopping callback
early_stopping_ftse100 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
ftse100_model.fit(ftse100_x_train, ftse100_y_train, epochs=25, batch_size=64, validation_split=0.1, callbacks=[early_stopping_ftse100])

# Prepare test data for FTSE 100
ftse100_x_test = []

for x in range(prediction_days, len(ftse100_scaled_data)):
    ftse100_x_test.append(ftse100_scaled_data[x - prediction_days:x, 0])

ftse100_x_test = np.array(ftse100_x_test)
ftse100_x_test = np.reshape(ftse100_x_test, (ftse100_x_test.shape[0], ftse100_x_test.shape[1], 1))

# Plot predictions for S&P 500 and FTSE 100 together
plot_stock_predictions_multi([sp500_model, ftse100_model], [sp500_scaler, ftse100_scaler], [sp500_data, ftse100_data], [sp500_x_test, ftse100_x_test], start_date, end_date, prediction_days=60, stock_symbols=['S&P 500', 'FTSE 100'])
