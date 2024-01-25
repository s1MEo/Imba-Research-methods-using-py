import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Get user input for company symbol
company = input("Enter the stock symbol of the company you want to analyze (e.g., AMZN): ").upper()

# Define the time period
start = dt.datetime(2022, 1, 1)
end = dt.datetime.now()

# Download historical stock data using yfinance
data = yf.download(company, start=start, end=end)

# Check if data is available for the given company
if data.empty:
    print(f"No data available for the company with symbol {company}. Please check the symbol.")
else:
    # Prepare the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prediction days
    prediction_days = 60

    # Prepare the training data
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Introduce EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with early stopping
    model.fit(x_train, y_train, epochs=25, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

    # Test the model accuracy on existing data
    test_start = dt.datetime(2023, 1, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(company, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    # Combine training and test datasets for plotting
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make prediction on Test data
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the training, test, and predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, scaler.inverse_transform(scaled_data), color='gray', label="Training Data")
    plt.plot(test_data.index, actual_prices, color='blue', label=f"Actual {company} Price (Test Period)")
    plt.plot(test_data.index, predicted_prices, color='green', label=f"Predicted {company} Price (Test Period)")
    plt.axvline(test_start, color='red', linestyle='--', linewidth=2, label='Test Start')
    plt.axvline(test_end, color='red', linestyle='--', linewidth=2, label='Test End')
    plt.title(f"{company} Share Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()

    # Predicting the next day
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs)+1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction price of {company} tomorrow is: ${prediction[0][0]:.2f}")
