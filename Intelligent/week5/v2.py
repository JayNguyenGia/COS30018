# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

from msilib import Feature
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
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

#Task3-
def plot_boxplot(df, n, columns):
    # Calculate the rolling window data for each column
    rolling_data = [df[column].rolling(n).mean() for column in columns]
    
    # Create the box plot
    fig, ax = plt.subplots()
    ax.boxplot([data.dropna() for data in rolling_data], labels=columns)
    ax.set_title(f'{n} Day Rolling Window')
    
    # Show the plot
    plt.show()
    
def plot_candlestick(df, n=1):
    #Resample the data
    df = df.resample(f'{n}D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    mpf.plot(df,type='candle')# Create candle chart
    
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

#Task2-Functionn to load and process
def processData(
    ticker, #company ticker symbol
    start_date, #day start
    end_date, # day end
    save_file, # whether to save dataset to a file
    prediction_column, # 
    prediction_days, # number of day predict in the future
    feature_columns=[], # list of features coolumns
    split_method='date', # method to split to train or test 
    split_ratio=0.8, # radio of train/test if the spit method is 'random'
    split_date=None, # date to split the data
    fillna_method='drop', # method to drop or fill NaN values 
    scale_features=False, # 
    scale_min=0, # minimum value to scale the feature columns
    scale_max=1, # maximum value to scale the feature columns
    save_scalers=False):

    data = downloadData(ticker, start_date, end_date, save_file)
    
    result = {}
    # we will also return the original dataframe itself
    result['df'] = data.copy()
   
    # make sure that the passed feature_columns exist in the dataframe
    if len(feature_columns) > 0:
        for col in feature_columns:
            assert col in data.columns, f"'{col}' does not exist in the dataframe."
    else:
        # if no feature_columns are passed, use all columns except the prediction_column
        feature_columns = list(filter(lambda column: column != 'Date', data.columns))
    
    # add feature columns to result
    result['feature_columns'] = feature_columns
    # Deal with potential NaN values in the data
    # Drop NaN values
    data = processNANs(data, fillna_method)

    # Split data into train and test sets based on date
    if split_method == 'date':
        train_data = data.loc[data['Date'] < split_date]
        test_data = data.loc[data['Date'] >= split_date]
    # Split data into train and test sets randomly with provided ratio
    elif split_method == 'random':
        train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
    # Reset index of both dataframes
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()
    # Sort dataframes by date
    train_data = train_data.sort_values(by='Date')
    test_data = test_data.sort_values(by='Date')

    # Scale features
    if scale_features:
        # Create scaler dictionary to store all scalers for each feature column
        scaler_dict = {}
        # Dictionaries to store scaled train and test data
        scaled_train_data = {}
        scaled_test_data = {}
        #loop through each feature column
        for col in feature_columns:
            # Create scaler for each feature column using Min Max, passing in the scale_min and scale_max
            scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
            # Fit and transform scaler on train data
            scaled_train_data[col] = scaler.fit_transform(train_data[col].values.reshape(-1, 1)).ravel()
            # Transform test data using scaler
            scaled_test_data[col] = scaler.transform(test_data[col].values.reshape(-1,1)).ravel()
            # Add scaler to scaler dictionary, using the feature column name as key
            scaler_dict[col] = scaler
        # Add scaler dictionary to result
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
       
        # Convert scaled data to dataframes
        train_data = pd.DataFrame(scaled_train_data)
        test_data = pd.DataFrame(scaled_test_data)

    # Add train and test data to result
    result["scaled_train"] = train_data
    result["scaled_test"] = test_data
    # Construct the X's and y's for the training data
    X_train, y_train = [], []
    # Loop through the training data from prediction_days to the end
    for x in range(prediction_days, len(train_data)):
        # Append the values of the passed prediction column to X_train and y_train
        X_train.append(train_data[prediction_column].iloc[x-prediction_days:x])
        y_train.append(train_data[prediction_column].iloc[x])

    # convert to numpy arrays
    result["X_train"] = np.array(X_train)
    result["y_train"] = np.array(y_train)
    # reshape X_train for proper fitting into LSTM model
    result["X_train"] = np.reshape(result["X_train"], (result["X_train"].shape[0], result['X_train'].shape[1], -1));
    # construct the X's and y's for the test data
    X_test, y_test = [], []
    # Loop through the test data from prediction_days to the end
    for x in range(prediction_days, len(test_data)):
        # Append the values of the passed prediction column to X_test and y_test
        X_test.append(test_data[prediction_column].iloc[x - prediction_days:x])
        y_test.append(test_data[prediction_column].iloc[x])

    # convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #assign y_test to result
    result["y_test"] = y_test
    #assign X_test to result and reshape X_test for prediction compatibility
    result["X_test"] = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1));

    return result
# Load ad process data using Task B.2
# Function parameters to use
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
    save_scalers=SAVE_SCALERS
    )

plot_candlestick(processNANs(downloadData(COMPANY, '2022-05-01', '2022-05-31', False), 'drop'), 5)

#plot_boxplot(processNANs(downloadData(COMPANY, '2019-01-01', '2022-12-31', False),'drop'),['Open', 'High', 'Low', 'Close', 'Adj Close'], 10)

plot_boxplot(downloadData(COMPANY, '2019-01-01', '2022-12-31', False), 40, ['Open', 'High', 'Low', 'Close', 'Adj Close'])

model = Sequential() # Basic neural network

model.add(LSTM(units=50, return_sequences=True, input_shape=(data["X_train"].shape[1], 1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')


# Now we are going to train this model with our training data 
# (x_train, y_train)
#model.fit(x_train, y_train, epochs=25, batch_size=32)
model.fit(data['X_train'], data["y_train"], epochs=25, batch_size=32)

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
#TEST_START = '2023-08-02'
#TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

#test_data = yf.download(COMPANY,TEST_START,TEST_END)


# The above bug is the reason for the following line of code
# test_data = test_data[1:]

#actual_prices = test_data[PRICE_VALUE].values
actual_prices = data["column_scaler"][prediction_column].inverse_transform(data["y_test"].reshape(-1,1))
#total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

#model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

#model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

#model_inputs = scaler.transform(model_inputs)


#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
#x_test = []
#for x in range(PREDICTION_DAYS, len(model_inputs)):
#    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

#x_test = np.array(x_test)
#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

#predicted_prices = model.predict(x_test)
predicted_prices = model.predict(data['X_test'])
#predicted_prices = scaler.inverse_transform(predicted_prices)
predicted_prices = data["column_scaler"][prediction_column].inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

predicted_prices = predicted_prices.ravel()
actual_prices = actual_prices.ravel()
df = pd.DataFrame(predicted_prices)
df.to_csv('predicted_prices.csv', index=False)
df = pd.DataFrame(actual_prices)
df.to_csv('actual_prices.csv', index=False)
#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


#real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = [data['X_test'][len(data['X_test']) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
#prediction = scaler.inverse_transform(prediction)
prediction = data["column_scaler"][prediction_column].inverse_transform(prediction)
print(f"Prediction: {prediction[0]}")

