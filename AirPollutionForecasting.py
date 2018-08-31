import numpy as np


# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # This forces Keras to use CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense, CuDNNLSTM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
from keras.callbacks import TensorBoard


# Verify GPU is being utilized
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) # displays what you're using



"""
------------------------------------------------------------------------------
TIMING:
Batchsize = 
- LSTM on GPU took around 170 sec avg to train 50 epochs
- CuDNNLSTM on GPU took around 120 sec on avg to train 50 epochs
- LSTM on CPU took around 20 seconds to train 50 epochs

Someone on stack overflow suggested that I increase the batch size. This seems to happen occasionally with LSTMs (i.e. GPU time > CPU time)

- Sure enough, increasing the batch size to 512 reduced training time:
    - LSTM on GPU took ~ 35 sec to train 50 epochs
    - CuDNNLSTM on GPU took ~ 13 seconds for 50 epochs
    - CPU time with this batch size takes 8 seconds for 50 epochs...

    Why does increasing batch size make GPU work faster?


    - Maybe it's the scale of the data we're processing. 
       

------------------------------------------------------------------------------

Use the tutorial as a baseline for accuracy.
Things to play around with: 

    BE SURE TO USE SAME RANDOM SEED FOR COMPARING PERFORMANCE!
    
    MIGHT BE BETTER TO DO THIS ON YOUR OWN PROJECT SO AS TO PUBLISH ORIGINAL WORK 

    Preprocessing:
        - considering more than 1 hour in the past for input 
            --> try various values and see at what point it diminishes. It's pretty straightforward that if you consider too much information from the past to predict weather in the near future, it will decrease performance.
                 This is simply because weather that happened 3 days ago is typically not relevant in predicting what the weather will be like in the next few hours. This range will be different for various time-series problems.
        - making data stationary
        - using one-hot vectors for wind?? maybe more variables
        
        
    Model Architecture:
        - Try various sizes for the memory cell and see how that impacts performance.
        
        - Hyperparameter tuning
        
        - Compare GRU vs. LSTM vs. RNN
        
        - Various optimization techniques 
        
"""


# Following a tutorial on machine learning mastery
def parse_dates(x):
    return datetime.strptime(x, '%Y %m %d %H')

# pd.read_csv has a parameter which allows use to parse certain columns as a datetime object
# parse_dates - integers or strings which indicate which columns contain date information to be used in converting to datetime obj
# date_parser - function which converts the elements of the columns provided in 'parse_dates' into datetime objects

# weather_data = pd.read_csv('H:\Summer Research 2018\IntroToTensorFlow\PRSA_data_2010.1.1-2014.12.31.csv', parse_dates = [['year', 'month', 'day', 'hour']], index_col= 0, date_parser = parse_dates)
weather_data = pd.read_csv('/Volumes/x2016onq$/Summer Research 2018/IntroToTensorFlow/PRSA_data_2010.1.1-2014.12.31.csv', parse_dates = [['year', 'month', 'day', 'hour']], index_col= 0, date_parser = parse_dates)
print(type(weather_data))
print(weather_data.head())
weather_data.drop(columns = ['No'], inplace = True)  # for some reason this doesn't run on mac

weather_data.columns = ['Pollution', 'Dew', 'Temp', 'Press', 'Wind Direction', 'Wind Speed', 'Snow', 'Rain']
weather_data.index.name = 'Date'
weather_data['Pollution'].fillna(0, inplace=True)   # set the na values of Pollution = 0
weather_data = weather_data[24:]                    # omitting the first 24 hours of data

print(weather_data.head(5))
assert not weather_data.isnull().values.any()   # isnull() returns boolean Pandas Series, .values -> converts this Series into a numpy array, .any() returns true if any values are null in this case
# weather_data.to_csv('H:\Summer Research 2018\IntroToTensorFlow\PollutionData.csv')
# print('years of data = ', weather_data.shape[0] // (24 * 365))
groups = list(range(8))
values = weather_data.values
# print('values = ', values)
# plot each series of the dataframe as a separate subplot
plt.figure()
i = 1
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])           # plot each col of df
    plt.title(weather_data.columns[group], y = 0.5, loc = 'right')
    i += 1
# plt.show()
plt.close()
# TODO: Predict the pollution for the next hour based on the weather conditions and pollution over the last 24 hours.
# TODO: Predict the pollution for the next hour as above and given the “expected” weather conditions for the next hour.

def series_to_supervised(data, n_in = 1, n_out = 1, dropnan = True):
    """
    :param data: time series data
    :param n_in: how many previous time steps we want to consider
    :param n_out: how many timesteps into the future we want in our aggregated df
    :param dropnan: bool - dictates whether we will drop NaN values
    :return: aggregated df containing data from the last n_in timesteps
    """
    n_vars = 1 if type(data) is list else data.shape[1]                 # if the data is a list, there's only one feature var, otherwise the number of features is equal to the number of cols in df
    df = pd.DataFrame(data)
    cols, names = [], []                # list to store column values and their respective names
    for i in range(n_in, 0, -1):        # generate the input sequence to fed into LSTM
        cols.append(df.shift(i))        # add the shifted values of the dataframe, shift(1) will shift values down by 1
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]            # %d is placeholder for a number in python string interpolation

    for i in range(0, n_out):
        cols.append(df.shift(-i))           # shift rows up by i
        if i == 0:
            names += [('var%d(t)') % (j+1) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)') % (j+1, i) for j in range(n_vars)]

    agg = pd.concat(cols, axis= 1)          # aggregate all columns into aingular data frame. cols is a list of dataframes. each df holds col values for each col in data. axis = 1, specifies we want to aggregate along cols
    agg.columns = names                     # adding column names

    if dropnan:
        agg.dropna(inplace=True)
    return agg

encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])  # encoding ranks the numbers in ascending order. 0 -> smallest number, num_elements -> largest number. If two numbers have same value they are encoded identically
values = values.astype('float32')       # type cast values, np version of df, to all float32
# print("VALUES = ", values)

# ASIDE: LABELENCODER()
# e.g. en.fit_transform([100,10, 250, 30])   # where en = LabelEncoder()
# Out[18]: array([2, 0, 3, 1], dtype=int64)
# en.fit_transform([100,100, 250, 30])
# Out[19]: array([1, 1, 2, 0], dtype=int64)

scaler = MinMaxScaler(feature_range= (0, 1))        # MinMaxScaler normalizes inputs to be in the range [0,1]
scaled = scaler.fit_transform(values)               # normalizing the values and returning them
super_data = series_to_supervised(scaled, 1, 1)     # reframe as supervised learning data


# Do we only want to consider var1 for each future timestep?????? ---------------------- NEED TO CHANGE THE COLS DROPPED IF FORECASTING MORE THAN 1 TIMESTEP INTO THE FUTURE
dropped_cols = list(range(super_data.shape[1]-7, super_data.shape[1]))       # drop the last 7 cols, we only want to forecast var1(t), not var1(t) and var2(t) and var3(t) ...
super_data.drop(super_data.columns[dropped_cols], axis = 1, inplace= True)   # drop extraneous columns
print(super_data.head())

#------- Splitting up Data ------
values = super_data.values  # numpy array of df
num_years_training = 4
n_train_hrs = 365 * num_years_training * 24
train = values[:n_train_hrs,:]          # np array
test = values[n_train_hrs:, :]          # np array
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])        # reshape data into 3D array, which is expected input format of Keras LSTM model: (num_examples, num_timesteps for each example, num features)
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# ------- Keras LSTM Model ---------

# What does it mean that the internal state of the LSTM is reset at the end of each batch?

# Answer:


# # TRAIN MODEL
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# TENSORBOARD
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))


# fit network
start = time.time()
history = model.fit(train_X, train_y, epochs=50, batch_size=256, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks = [tensorboard])
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
print('training time = ', time.time() - start)



# ------- TRAINING CuDNNLSTM ---------      # training time was about 50 seconds faster than the regular lstm (for 50 epochs)
# model = Sequential()
# model.add(CuDNNLSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(1))
# model.compile(loss = 'mae', optimizer = 'adam')
# start = time.time()
# history = model.fit(train_X, train_y, epochs=50, batch_size=512, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# print("training time = ", time.time() - start)
# plt.plot(history.history['loss'], label = 'train')
# plt.plot(history.history['val_loss'], label = 'test')
# plt.legend()
# plt.show()



# EVALUATE MODEL
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



