# lstm for time series forecasting
from numpy import sqrt, asarray
import pandas as pd
from pandas import read_csv
from tensorflow import keras, math, reduce_mean
from keras import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import time
import settings
from datetime import datetime
import sys
import random

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
def split_sequence(sequence, n_steps=5):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        # filter this properly
        # TODO: training on an irregular matrix
        seq_x, seq_y = sequence[i:end_ix+1, :], sequence[end_ix, -1]
        seq_x[-1, -1] = 0
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

def generate_model(starting, model_location, csv):
   df = read_csv(csv, index_col=0)

   df = df.drop(['year', 'day'], axis=1)
   df_predictor = asarray(df.iloc[starting:starting + N_STEPS + 1]).reshape(1, N_STEPS + 1, SET_SIZE)
   dataframeset = [df.iloc[:starting], df.iloc[starting + N_STEPS + 1:]]
   df = pd.concat(dataframeset)

   # retrieve the values
   values = df.values.astype('float32')
   # specify the window size
   # split into samples
   X, y = split_sequence(values, N_STEPS)
   # reshape into [samples, timesteps, features]
   X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
   # split into train/test
   n_test = int(X.shape[0] / TEST_PROPORTION)
   X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

   callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
   model = Sequential()
   model.add(BatchNormalization(input_shape=(N_STEPS + 1, len(df.iloc[0]))))
   # LSTM to weight recency
   model.add(LSTM(100, activation='relu', kernel_initializer='he_normal'))
   # dense layers to regenerate spatiality
   model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
   model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
   model.add(Dense(1, activation='tanh'))
   # compile the model
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])

   # fit the model
   model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(X_test, y_test),
             callbacks=[callback])
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
    model.compile(optimizer='adam', loss='mse', metrics=['mae'], run_eagerly=True)
    little_x = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=VERBOSE, validation_data=(x_test, y_test))
    little_x.model.save(model_location)
    return x_test, y_test, little_x


def compute_prediction(model_location, df):
    global PREVIOUS_PREDICTION
    values = (df.loc[:,'value'].values).tolist()
    current = values[len(values) - 1]
    th = []
    current_amount = wait_amount(model_location, False, True)
    update = (current_amount == NEW_DATA_AMOUNT - 1)
    predict_model = keras.models.load_model(model_location)#this is copy that will be used to make predictions
    for i in range(settings.SUPPLY_TIME_HORIZONS):
        row = asarray(values[-N_STEPS:]).reshape((1, N_STEPS, 1))
        th.append(predict_model.predict(row, verbose=VERBOSE))
        values.append(th[i][0][0])
    accuracy = 1
    # Update if batch size reached or predictions become inaccurate
    if (PREVIOUS_PREDICTION == SUPPLY_UNINIT):
        PREVIOUS_PREDICTION = th[0][0][0]
    else:
        accuracy = abs((PREVIOUS_PREDICTION - current) / current)
        PREVIOUS_PREDICTION = th[0][0][0]

    # current_amount = wait_amount(model_location, False, True)
    # Update if batch size reached or predictions become inaccurate
    if (COUNTER == (NEW_DATA_AMOUNT - 1)) or ((accuracy < PREDICTION_THRESHOLD) and (wait_amount(model_location, False, False) > (2 * 5))):
        # validate on past 5 hours of data for refitting model
        wait_amount(model_location, True, False)
        fit_model(predict_model, df, current_amount, model_location, 5)

    return ([current] + [th[i][0][0] for i in range(settings.SUPPLY_TIME_HORIZONS)])