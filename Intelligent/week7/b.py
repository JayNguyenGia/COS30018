# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Further iterations by Naca Hitchman

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

from msilib import Feature
import string
from xml.etree.ElementTree import tostring
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import os
import pickle
import csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dropout, Dense, ConvLSTM2D, TimeDistributed, Flatten, Lambda
from keras.layers import Reshape

import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pmdarima import auto_arima
from pandas.plotting import autocorrelation_plot



#task 5
#   multi step prediction function
def create_sequences_ms(data, seq_length, n_steps_ahead):
    xs = []
    ys = []
    for i in range(len(data)-(seq_length+n_steps_ahead)+1):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length):(i+seq_length+n_steps_ahead)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_sequences_mv(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-(seq_length+1)):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_sequences(data, n_inputdays, n_outputdays):
    X = []
    y = []
    for i in range(len(data) - n_inputdays - n_outputdays + 1):
        X.append(data[i:(i + n_inputdays)])
        y.append(data[(i + n_inputdays):(i + n_inputdays + n_outputdays)])
    return np.array(X), np.array(y)

#task 4
def create_model(n_inputdays, n_features, n_outputdays, cell_types=["LSTM"], n_neurons=50, dropout_rate=0.2, loss="mse", optimizer="adam"):
    cell_type_map = {
        "LSTM": LSTM,
        "GRU": GRU,
        "SimpleRNN": SimpleRNN
    }
    
    for cell_type in cell_types:
        assert cell_type in cell_type_map, f"Invalid cell_type: {cell_type}. Choose from {list(cell_type_map.keys())}"
    
    model = Sequential()
    
    for i, cell_type in enumerate(cell_types):
        CellLayer = cell_type_map[cell_type]
        return_sequences = i != len(cell_types) - 1 or n_outputdays > 1  # Only the last LSTM layer should return sequences if n_outputdays > 1
        if i == 0:
            model.add(Bidirectional(CellLayer(n_neurons[i], activation='relu', return_sequences=return_sequences), input_shape=(n_inputdays, n_features)))
        else:
            model.add(Bidirectional(CellLayer(n_neurons[i], activation='relu', return_sequences=return_sequences)))
        model.add(Dropout(dropout_rate))
    
    # Apply a TimeDistributed Dense layer to each time step independently
    model.add(TimeDistributed(Dense(n_features)))
    # Select the last n_outputdays time steps from the sequence
    model.add(Lambda(lambda x: x[:, -n_outputdays:, :]))
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

#task 3
def plot_boxplot(df, n, columns):
    # Calculate the rolling window data for each column
    rolling_data = [df[column].rolling(n).mean() for column in columns]
    
    # Create the box plot
    fig, ax = plt.subplots()
    ax.boxplot([data.dropna() for data in rolling_data], labels=columns)
    ax.set_title(f'{n} Day Rolling Window')
    
    # Show the plot
    plt.show()

#task 3
def plot_candlestick(df, n=1):
    # Resample the data to have one row per n trading days
    df = df.resample(f'{n}D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    # Create the candlestick chart
    mpf.plot(df, type='candle')

#task 2
def downloadData(ticker, start_date, end_date, save_file=False):
     #create data folder in working directory if it doesnt already exist
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data = None 
    #if ticker is a string, load it from yfinance library
    if isinstance(ticker, str):
        # Check if data file exists based on ticker, start_date and end_date
        file_path = os.path.join(data_dir, f"{ticker}_{start_date}_{end_date}.csv")
        if os.path.exists(file_path):
            # Load data from file
            data = pd.read_csv(file_path)
        else:
            # Download data using yfinance
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Save data to file if boolean save_file is True
            if save_file:
                data.to_csv(file_path) 
    #if passed in ticker is a dataframe, use it directly
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        data = ticker
    else:
        # raise error if ticker is neither a string nor a dataframe
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # return the dataframe
    return data

def processNANs(df, fillna_method):
    # Deal with potential NaN values in the data
    # Drop NaN values
    if fillna_method == 'drop':
        df.dropna(inplace=True)
    #use forward fill method, fill NaN values with the previous value
    elif fillna_method == 'ffill':
        df.fillna(method='ffill', inplace=True)
    #use backward fill method, fill NaN values with the next value
    elif fillna_method == 'bfill':
        df.fillna(method='bfill', inplace=True)
    #use mean method, fill NaN values with the mean of the column
    elif fillna_method == 'mean':
        df.fillna(data.mean(), inplace=True)

    return df
#task 2 - function to load and process a dataset with multiple features
def processData(ticker, start_date, end_date, save_file, prediction_column, prediction_days, feature_columns=[], split_method='date', split_ratio=0.8, split_date=None, fillna_method='drop', scale_features=False, scale_min=0, scale_max=1, save_scalers=False, n_steps=5):
    data = downloadData(ticker, start_date, end_date, save_file)
    result = {'df': data.copy()}
    
    if not feature_columns:
        feature_columns = [col for col in data.columns if col != 'Date']
    
    result['feature_columns'] = feature_columns
    data = processNANs(data, fillna_method)

    if split_method == 'date':
        train_data = data[data['Date'] < split_date]
        test_data = data[data['Date'] >= split_date]
    elif split_method == 'random':
        train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
    train_data = train_data.sort_values(by='Date').reset_index(drop=True)
    test_data = test_data.sort_values(by='Date').reset_index(drop=True)
    
    result["train_data_unscaled"] = train_data
    result["test_data_unscaled"] = test_data

    if scale_features:
        scaler_dict = {}
        for col in feature_columns:
            scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
            train_data[col] = scaler.fit_transform(train_data[col].values.reshape(-1, 1))
            test_data[col] = scaler.transform(test_data[col].values.reshape(-1, 1))
            scaler_dict[col] = scaler
        result["column_scaler"] = scaler_dict
    
            # Save scalers to file
        if save_scalers:
            # Create scalers directory if it doesn't exist
            scalers_dir = os.path.join(os.getcwd(), 'scalers')
            if not os.path.exists(scalers_dir):
                os.makedirs(scalers_dir)
            # Create scaler file name
            scaler_file_name = f"{ticker}_{start_date}_{end_date}_scalers.txt"
            scaler_file_path = os.path.join(scalers_dir, scaler_file_name)
            with open(scaler_file_path, 'wb') as f:
                pickle.dump(scaler_dict, f)

    result["train_data"] = train_data
    result["test_data"] = test_data

    X_train, y_train = create_sequences(train_data[feature_columns].values, n_inputdays=PREDICTION_DAYS, n_outputdays=N_STEPS)
    X_test, y_test = create_sequences(test_data[feature_columns].values, n_inputdays=PREDICTION_DAYS, n_outputdays=N_STEPS) 

    result["X_train"] = X_train
    result["y_train"] = y_train
    result["X_test"] = X_test
    result["y_test"] = y_test

    return result

#------------------------------------------------------------------------------
# Load and process data using Task B.2 Function
# ------------------------------------------------------------------------------

# define function parameters to use
DATA_SOURCE = "yahoo"
COMPANY = "TSLA"  
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2022-12-31'
SAVE_FILE = True
PREDICTION_DAYS = 100
SPLIT_METHOD = 'random'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2020-01-02'
NAN_METHOD = 'drop'
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
SCALE_FEATURES = True
SCALE_MIN = 0
SCALE_MAX = 1
SAVE_SCALERS = True
prediction_column = "Close"
N_STEPS = 5;

# Call processData function passing in parameters
data = processData(
    ticker=COMPANY, 
    start_date=DATA_START_DATE, 
    end_date=DATA_END_DATE, 
    save_file=SAVE_FILE,
    prediction_column=prediction_column,
    prediction_days=PREDICTION_DAYS,
    split_method=SPLIT_METHOD, 
    split_ratio=SPLIT_RATIO, 
    split_date=SPLIT_DATE,
    fillna_method=NAN_METHOD,
    feature_columns=FEATURE_COLUMNS,
    scale_features=SCALE_FEATURES,
    scale_min=SCALE_MIN,
    scale_max=SCALE_MAX,
    save_scalers=SAVE_SCALERS,
    n_steps=N_STEPS
    )
#task 4 candlestick
#plot_candlestick(processNANs(downloadData(COMPANY, '2022-05-01', '2022-05-31', False), 'drop'), 5)

#plot_boxplot(processNANs(downloadData(COMPANY, '2019-01-01', '2022-12-31', False),'drop'),['Open', 'High', 'Low', 'Close', 'Adj Close'], 10)

#task 4 bloxplot
#plot_boxplot(downloadData(COMPANY, '2019-01-01', '2022-12-31', False), 40, ['Open', 'High', 'Low', 'Close', 'Adj Close'])
# Number of days to look back to base the prediction
#PREDICTION_DAYS = 60 # Original

#------------------------------------------------------------------------------
#Task 4
sequence_length = data['X_train'].shape[1]
n_features = data['X_train'].shape[2]
#set 1
"""
units = [128, 64, 32]
cells = ['LSTM', 'LSTM','LSTM']
n_layers = 3
dropout = 0.3
loss = "mean_absolute_error"
optimizer = "rmsprop"
bidirectional = True

# Set the number of epochs and batch size
epochs = 32
batch_size = 64
"""
units = [32, 16]
cells = ['LSTM','LSTM']
n_layers = 2
dropout = 0.3
loss = "mean_absolute_error"
optimizer = "rmsprop"
bidirectional = True

# Set the number of epochs and batch size
epochs = 10
batch_size = 32

# Create the model using the create_model function
model = create_model(n_inputdays=PREDICTION_DAYS, n_features=n_features, n_outputdays=N_STEPS, n_neurons=units, cell_types=cells,
                     dropout_rate=dropout, loss=loss, optimizer=optimizer)

# Train the model on the training data
print(data['X_train'].shape)
print(data['y_train'].shape)
model.fit(data['X_train'], data['y_train'], epochs=epochs, batch_size=batch_size)

#task 6 train arima model
#arima_model = train_arima_model(data["train_data"]['Close'], order = (1,1,0))

#autocorrelation_plot(data['train_data'][prediction_column])
#plt.show()




#plt.plot(test_data)
#plt.plot(predictions, color='red')
#plt.show()

closing_price_index = FEATURE_COLUMNS.index(prediction_column)

# Get the actual prices

actual_prices = data["column_scaler"][prediction_column].inverse_transform(data["y_test"][:, -1, closing_price_index].reshape(-1,1)).ravel()
# Predict the prices with ltsm model
predicted_prices = model.predict(data['X_test'])
predicted_close_prices = predicted_prices[:, -1, closing_price_index].reshape(-1, 1)
predicted_close_prices = data["column_scaler"][prediction_column].inverse_transform(predicted_close_prices).ravel()


test_data = data['test_data'][prediction_column].values[-len(predicted_close_prices):]
history = [x for x in data['train_data'][prediction_column].values]
"""
predictions = list()
for t in range(len(test_data)):
	arima_model = ARIMA(history, order=(5,1,0))
	model_fit = arima_model.fit()
	output = model_fit.forecast()
	forcast = output[0]
	predictions.append(forcast)
	obs = test_data[t]
	history.append(obs)
	print('%f/%f, predicted=%f, expected=%f' % (t,len(test_data), forcast, obs))
    
arima_predictions_scaled = data["column_scaler"][prediction_column].inverse_transform(np.array(predictions).reshape(-1,1)).reshape(-1)
ensemble_prediction = (predicted_close_prices + arima_predictions_scaled) / 2
#arima_predictions_scaled = arima_predictions_scaled[-len(predicted_close_prices):]
#task 6 ensemble prediction
# Ensure both arrays are numpy arrays
#predicted_close_prices = np.array(predicted_close_prices)
#arima_predictions_scaled = np.array(arima_predictions_scaled)

# Check if both arrays have the same shape
print (predicted_close_prices.shape)
print (arima_predictions_scaled.shape)
if predicted_close_prices.shape != arima_predictions_scaled.shape:
    raise ValueError("Both arrays must have the same shape to calculate the ensemble average")
"""
#sarima model
#sarima_model = SARIMAX(history, order=(5, 1, 0), seasonal_order=(1, 1, 0, 90))

    
#arima_predictions_scaled = data["column_scaler"][prediction_column].inverse_transform(np.array(predictions).reshape(-1,1)).reshape(-1)

#random forrest model

# Create a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=300, random_state=42)
#X_train_reshaped =  data["X_train"][:, :, closing_price_index]
X_train_flattened = data["X_train"][:, :, closing_price_index].reshape(data["X_train"].shape[0], -1)
#y_train_reshaped = data["y_train"].reshape(-1, 1)  # shape will be (1507, 100*6)
# Fit the model on your training data

y_train_reshaped = data["y_train"][:, :, closing_price_index]  # shape will be (1507, 5)
print(X_train_flattened.shape)
print(y_train_reshaped.shape)
rf.fit(X_train_flattened, y_train_reshaped)

# Make predictions on the test data
X_test_2d = data["X_test"].reshape((data["X_test"].shape[0], -1)) 
rf_predictions = rf.predict(data["X_test"][:, :, closing_price_index])

rf_predictions = data["column_scaler"][prediction_column].inverse_transform(np.array(rf_predictions).reshape(-1,1)).reshape(-1)
rf_predictions = rf_predictions[-len(predicted_close_prices):]
ensemble_prediction = (predicted_close_prices + rf_predictions) / 2
  
# Plot the actual and predicted prices
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_close_prices, color="green", label=f"Predicted {COMPANY} Price LTSM")

plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Save the predicted and actual prices to csv files
predicted_prices = predicted_close_prices.ravel()
actual_prices = actual_prices.ravel()
df = pd.DataFrame(predicted_close_prices)
df.to_csv('predicted_prices.csv', index=False)
df = pd.DataFrame(actual_prices)
df.to_csv('actual_prices.csv', index=False)

# Predict the next k days
real_data = [data['X_test'][-1, :, :]]
real_data = np.array(real_data)
print(real_data.shape)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], n_features))

# Predict the next k days
prediction = model.predict(real_data)  # shape: (1, k, n_features)

# Reshape the prediction array to be 2D
reshaped_prediction = prediction[:, :, closing_price_index].reshape(-1, 1)  # shape: (k, 1)

# Apply inverse transformation
prediction = data["column_scaler"][prediction_column].inverse_transform(reshaped_prediction)  # shape: (k, 1)

# Flatten the array to have shape (k,)
prediction = prediction.ravel()

arima_model = ARIMA(data['train_data'][prediction_column], order=(5,1,0))
model_fit = arima_model.fit()

arima_prediction = model_fit.forecast(steps=N_STEPS)
#arima_prediction = arima_prediction[0]
arima_prediction = data["column_scaler"][prediction_column].inverse_transform(np.array(arima_prediction).reshape(-1,1)).ravel()
ensemble_prediction = (prediction + arima_prediction) / 2


# Loop over the prediction and print each day's predicted price
for i, price in enumerate(prediction):
    print(f"LTSM Prediction for day {i+1}: {price}")
for i, price in enumerate(arima_prediction):
    print(f"ARIMA Prediction for day {i+1}: {price}")
for i, price in enumerate(ensemble_prediction):
    print(f"Ensemble Prediction for day {i+1}: {price}")