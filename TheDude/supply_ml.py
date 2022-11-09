from numpy import sqrt
import keras
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time
from datetime import datetime
from tensorflow import math
from tensorflow import reduce_mean
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
import settings
import glob
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

N_STEPS = 5
SET_SIZE = 19
TEST_PROPORTION = 6
NUM_DATA_POINTS = 1500 # MAX if using all data, integer if using some data
NUM_EPOCHS = 100
NEW_DATA_AMOUNT = 168
VERBOSE = 2
PREDICTION_THRESHOLD = .11 # Percentage
SUPPLY_UNINIT = 42069
COUNTER = 0
PREVIOUS_PREDICTION = 42069
PREDICTION_SET_SIZE = 100
N_TEST = 50
# Dictionary key is cluster model path, value is list with [prediction accuracy, counter]

# Should be used by layer above to increment amount of time horizons that ML has predicted
# rst set to true in order to reset after new data has been added into model
# inc set to true to increment counter, false to just return value of counter
def wait_amount(model_location, rst, inc):
    global COUNTER
    if rst: COUNTER = 0 #Set counter to 0
    elif inc: COUNTER += 1
    return COUNTER

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    print('X: \n', X[0:5], '\n Y: \n', y[0:5])
    return asarray(X), asarray(y)


def custom_loss(y_actual, y_pred):
    SE_Tensor = math.square(y_pred - y_actual)  # squared difference
    MSE = reduce_mean(SE_Tensor, axis=0)
    # RMSE = tf.math.sqrt(MSE)

    Zeros = tf.zeros_like(MSE)  # create tensor of zeros
    Mask = [False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True]  # create mask
    Solar_MSE = tf.where(Mask, MSE, Zeros)  # create tensor where every loss is 0 except solar output

    # print_output = tf.print(Solar_MSE, "Solar_MSE: ")

    return Solar_MSE


def custom_eval(y_actual, y_pred):
    SE_Tensor = math.square(y_pred - y_actual)  # squared difference
    MSE = reduce_mean(SE_Tensor, axis=0)

    Zeros = tf.zeros_like(MSE)  # create tensor of zeros
    Mask = [False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, True]  # create mask
    Solar_MSE = tf.where(Mask, MSE, Zeros)  # create tensor where every loss is 0 except solar output

    Solar_RMSE = tf.math.sqrt(Solar_MSE)

    # print_output = tf.print(Solar_RMSE, "Solar_RMSE: ")

    return Solar_RMSE

def generate_model(starting, model_location, csv):
    '''df = pd.read_csv(csv, index_col=0)
    df = df.head(n=starting)
    column_mean = []
    column_std = []
    pastWeatherAndSupply = df.iloc[:-2].drop(['year', 'day'], axis=1)

    currentWeather = df.iloc[-2]
    currentWeather = currentWeather.drop(['Percent Output', 'year', 'day'])  # dropping supply column for testing
    currentWeather = currentWeather.values[:].reshape((1, len(df.iloc[-2]) - 3))

    nextHourWeather = df.iloc[-1]
    nextHourWeather = nextHourWeather.drop(['Percent Output', 'year', 'day'])  # dropping supply column for testing
    nextHourWeather = nextHourWeather.values[:].reshape((1, len(df.iloc[-1]) - 3))
    X, y = pastWeatherAndSupply.values[:, :-1], pastWeatherAndSupply.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    n_features = X_train.shape[1]
    print(n_features)
    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_features,)))
    model.add(Dense(12, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(8, activation='selu', kernel_initializer='he_normal'))
    model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='tanh', kernel_initializer='he_normal'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=80, batch_size=10)
    model.save(model_location)'''
    df = read_csv(csv, index_col=0)
    df = df.head(n=starting)
    df_holdout = df.iloc[-PREDICTION_SET_SIZE:]
    df = df.iloc[:-PREDICTION_SET_SIZE]

    # holdout_X, holdout_y = split_sequence(df_holdout.values.astype('float32'), N_STEPS)
    # retrieve the values
    values = df.values.astype('float32')
    # # specify the window size
    # n_steps = 5
    # split into samples
    X, y = values[:, :-1], values[:, -1]
    X = X.reshape(X.shape[0],1,X.shape[1])
    # reshape into [samples, timesteps, features]
    # holdout_X = holdout_X.reshape((holdout_X.shape[0], holdout_X.shape[1], 21))
    # split into train/test
    # print(holdout_X[:-N_TEST])
    X_train, X_test, y_train, y_test = X[:-N_TEST], X[-N_TEST:], y[:-N_TEST], y[-N_TEST:]
    # %%
    # define model
    # improvement area : try adding dropout
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(1, 17)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    # model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    model.compile(optimizer='adam', loss='mse')
    # fit the model
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=2, validation_data=(X_test, y_test), callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3))
    model.save(model_location)

def fit_model(model, df, points, model_location, n_tests):
    df = df.tail(n=points)
    values = df.loc[:,'value'].values
    # split into samples
    X, y = split_sequence(values)
    # reshape into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # split into train/test
    x_train, x_test, y_train, y_test = X[:-n_tests], X[-n_tests:], y[:-n_tests], y[-n_tests:]
    # fit the model
    time1 = time.time()
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    little_x = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=VERBOSE, validation_data=(x_test, y_test))
    little_x.model.save(model_location)
    return x_test, y_test, little_x


def compute_prediction(model_location, df):

    # global PREVIOUS_PREDICTION
    # # values = (df.loc[:,'value'].values).tolist()
    current = df.tail(1)
    operate_current = current
    th = []
    # current_amount = wait_amount(model_location, False, True)
    # update = (current_amount == NEW_DATA_AMOUNT - 1)
    predict_model = keras.models.load_model(model_location, custom_objects={"custom_eval": custom_eval, "custom_loss": custom_loss})#this is copy that will be used to make predictions
    #TODO
    #This should be put into power world / Kara's thing later rather than calling .predict
    for i in range(settings.SUPPLY_TIME_HORIZONS):
        operate_current = asarray(df.iloc[100])[1:-1]
        operate_current = operate_current.reshape(operate_current.shape[1],1,17)
        # operate_current, y = split_sequence(df[-N_STEPS - 1:].values.astype('float32'), N_STEPS)
        # operate_current = operate_current.reshape((21, operate_current.shape[1], 21))
        # operate_current = df[-N_STEPS:].values.astype('float32')
        # x = list()
        # x.append(operate_current)
        prediction = predict_model.predict(operate_current, verbose=VERBOSE)
        # th.append(prediction)
        # df.append(prediction)
    # accuracy = 1
    # # Update if batch size reached or predictions become inaccurate
    # if (PREVIOUS_PREDICTION == SUPPLY_UNINIT):
    #     PREVIOUS_PREDICTION = th[0][0][0]
    # else:
    #     accuracy = abs((PREVIOUS_PREDICTION - current) / current)
    #     PREVIOUS_PREDICTION = th[0][0][0]
    #
    # # current_amount = wait_amount(model_location, False, True)
    # # Update if batch size reached or predictions become inaccurate
    # if (COUNTER == (NEW_DATA_AMOUNT - 1)) or ((accuracy < PREDICTION_THRESHOLD) and (wait_amount(model_location, False, False) > (2 * 5))):
    #     # validate on past 5 hours of data for refitting model
    #     wait_amount(model_location, True, False)
    #     fit_model(predict_model, df, current_amount, model_location, 5)
    # final_array = ([current] + [th[i][0][0] for i in range(settings.SUPPLY_TIME_HORIZONS)])
    return prediction

'''    currentWeather = df.iloc[-2]
    currentWeather = currentWeather.drop(['Percent Output', 'year', 'day'])  # dropping supply column for testing
    currentWeather = currentWeather.values[:].reshape((1, len(df.iloc[-2]) - 3))

    nextHourWeather = df.iloc[-1]
    nextHourWeather = nextHourWeather.drop(['Percent Output', 'year', 'day'])  # dropping supply column for testing
    nextHourWeather = nextHourWeather.values[:].reshape((1, len(df.iloc[-1]) - 3))
    model = keras.models.load_model(model_location, custom_objects={"custom_eval": custom_eval, "custom_loss": custom_loss})#this is copy that will be used to make predictions
    current_prediction = model.predict(currentWeather)
    if (current_prediction < 0):
        current_prediction = 0
    print('Predicted: %.3f' % current_prediction)
    print('Real: %.3f' % df.iloc[-2][-1])
    print(abs(df.iloc[-2][-1] - current_prediction) / df.iloc[-2][-1])

    future_prediction = model.predict(nextHourWeather)
    if (future_prediction < 0):
        future_prediction = 0
    print('Predicted: %.3f' % future_prediction)
    print('Real: %.3f' % df.iloc[-1][-1])
    print(abs(df.iloc[-1][-1] - future_prediction) / df.iloc[-1][-1])'''

# compute_prediction(settings.modelfp + "Supply", read_csv((glob.glob(settings.supplyfp)[0]), header=0, index_col=0).head(100))


