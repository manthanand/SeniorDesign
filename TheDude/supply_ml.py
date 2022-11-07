# lstm for time series forecasting
from numpy import sqrt, asarray
import pandas as pd
from pandas import read_csv
import tensorflow as tf
from tensorflow import keras, math, reduce_mean
from keras import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import time
import settings
from datetime import datetime
import sys
import random

N_STEPS = 5
SET_SIZE = 21
TEST_PROPORTION = 10
NUM_DATA_POINTS = 1500 # MAX if using all data, integer if using some data
NUM_EPOCHS = 60
NEW_DATA_AMOUNT = 168
VERBOSE = 2
PREDICTION_THRESHOLD = .11 # Percentage
SUPPLY_UNINIT = 42069
COUNTER = 0
PREVIOUS_PREDICTION = 42069

TIME = []
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
    return asarray(X), asarray(y)

def custom_loss(y_actual, y_pred):
    SE_Tensor = math.square(y_pred - y_actual)  #squared difference
    MSE = reduce_mean(SE_Tensor, axis=0)
    #RMSE = tf.math.sqrt(MSE)
    
    Zeros = tf.zeros_like(MSE) #create tensor of zeros
    Mask = [False] * SET_SIZE
    Mask[-1] = True

    Solar_MSE = tf.where(Mask, MSE, Zeros) #create tensor where every loss is 0 except solar output
    
    #print_output = tf.print(Solar_MSE, "Solar_MSE: ")
    
    return Solar_MSE

def custom_eval(y_actual, y_pred):
    SE_Tensor = math.square(y_pred - y_actual)  #squared difference
    MSE = reduce_mean(SE_Tensor, axis=0)
    
    Zeros = tf.zeros_like(MSE) #create tensor of zeros
    Mask = [False] * SET_SIZE
    Mask[-1] = True

    Solar_MSE = tf.where(Mask, MSE, Zeros) #create tensor where every loss is 0 except solar output
    Solar_RMSE = tf.math.sqrt(Solar_MSE)
    
    #print_output = tf.print(Solar_RMSE, "Solar_RMSE: ")
    
    return Solar_RMSE

def fit_model(model, df, points, model_location):
    df = df.tail(n=points)
    values = df.loc[:].values.astype('float32')
    # split into samples
    X, y = split_sequence(values)
    # reshape into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], SET_SIZE))
    n_test = int(X.shape[0] / TEST_PROPORTION)
    # split into train/test
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
    # fit the model
    time1 = time.time()
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    little_X = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=VERBOSE, validation_data=(X_test, y_test))
    TIME.append(time.time() - time1)
    little_X.model.save(model_location)
    return X_test, y_test, little_X

def generate_model(df, model_location):
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(N_STEPS,SET_SIZE)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])

    # fit the model
    x,y,model = fit_model(model, df, len(df) if (NUM_DATA_POINTS == "MAX") else NUM_DATA_POINTS, settings.supplyfp, int(df.shape[0] / TEST_PROPORTION))
    model.save(model_location)

def compute_prediction(model_location, df):
    global PREVIOUS_PREDICTION
    # reading list of values
    values = (df.loc[:,'value'].values.astype('float32')).tolist()
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