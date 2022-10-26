import pandas as pd
from datetime import datetime
import math
import os
from numpy import concatenate
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
import keras
import matplotlib.pyplot as plt
import csv
import jinja2
import settings
import time

NUM_DATA_POINTS = "MAX" # MAX if using all data, integer if using some data
NUM_EPOCHS = 100
N_STEPS = 5
NEW_DATA_AMOUNT = 168
VERBOSE = 2
PREDICTION_THRESHOLD = .04 # Percentage
DEMAND_UNINIT = 42069
# Dictionary key is cluster model path, value is list with [prediction accuracy, counter]
cluster_predictions = {}
TIME = []

#compiles list of new demand data to refit on
#ignores NULL elements
def get_new_data(old_data, amount):
    data = []
    for i in range(-1*amount, -1):
        if old_data[i] != -1:
            data.append(old_data[i])
    return data

# Should be used by layer above to increment amount of time horizons that ML has predicted
# rst set to true in order to reset after new data has been added into model
# inc set to true to increment counter, false to just return value of counter
def wait_amount(model_location, rst, inc):
    if rst: cluster_predictions[model_location][1] = 0 #Set counter to 0
    elif inc: cluster_predictions[model_location][1] += 1
    return cluster_predictions[model_location][1]

# split a univariate sequence into samples
def split_sequence(sequence):
    x, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + N_STEPS
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return asarray(x), asarray(y)

# This function fits the 'model' using an input pandas 'df' and the number of 'points' to fit on.
# 'points' is last number of points collected
# It also stores the fitted model in the filepath provided by 'model_location'
def fit_model(model, df, points, model_location, n_tests):
    df = df.tail(n=points)
    values = df.loc[:,'value'].values
    values = get_new_data(values, points)
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
    TIME.append(time.time() - time1)
    little_x.model.save(model_location)
    return x_test, y_test, little_x

# This function generates a demand model for LSTM given a dataframe with column parameter value
# that contains the data for amount of power used per time horizon. It assumes that the time
# between each data point is constant and returns the model.
def generate_model(df, model_location):
    # define model
    cluster_predictions[model_location] = [DEMAND_UNINIT, 0] #initialize all models [accuracy, counter]
    if (not os.path.exists(model_location)):
        model = Sequential()
        model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(N_STEPS, 1)))
        # model.add(BatchNormalization())
        model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1))
        # compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # Validate on last week of data for generating entire model
        x_test, y_test, model = fit_model(model, df, len(df) if (NUM_DATA_POINTS == "MAX") else NUM_DATA_POINTS, model_location, 150)

# This function generates a prediction given an input model and dataframe that contains the power consumption
# values in the value column. It will generate predictions for DEMAND_TIME_HORIZONS and update the model passed
# in with the new data point in the dataframe.
def compute_prediction(model_location, df):
    values = (df.loc[:,'value'].values).tolist()
    current = values[len(values) - 1]
    th = []
    current_amount = wait_amount(model_location, False, True)
    update = (current_amount == NEW_DATA_AMOUNT - 1)
    predict_model = keras.models.load_model(model_location)#this is copy that will be used to make predictions
    time1 = time.time()
    for i in range(settings.DEMAND_TIME_HORIZONS):
        row = asarray(values[-N_STEPS:]).reshape((1, N_STEPS, 1))
        th.append(predict_model.predict(row))
        values.append(th[i][0][0])
    accuracy = 1
    total_time = time.time() - time1
    # Update if batch size reached or predictions become inaccurate
    if (cluster_predictions[model_location][0] == DEMAND_UNINIT):
        cluster_predictions[model_location][0] = th[0][0][0]
    else:
        accuracy = abs((cluster_predictions[model_location][0] - current) / current)
    # current_amount = wait_amount(model_location, False, True)
    # Update if batch size reached or predictions become inaccurate
    if (cluster_predictions[model_location][1] == (NEW_DATA_AMOUNT - 1)) or ((accuracy < PREDICTION_THRESHOLD) and (wait_amount(model_location, False, False) > (2 * 5))):
        # validate on past 5 hours of data for refitting model
        wait_amount(model_location, True, False)
        fit_model(predict_model, df, current_amount, model_location, 5)
    else:
        TIME.append(total_time)

    return ([current] + [th[i][0][0] for i in range(settings.DEMAND_TIME_HORIZONS)])

def accuracy(last_15min_predication, index):
    try:
        actual_result = sum(index)
    except:
        actual_result = index
    acc = 100 - abs((actual_result - last_15min_predication) / actual_result * 100)
    return acc

def test_demonstration():
    CSV = "BuildingData2018_processed/ADH_E_TBU_CD_1514786400000_1535778000000_hourly.csv"
    j = 0
    demand_data = pd.read_csv(CSV)
    values = demand_data.loc[:, 'value'].values
    dates = []
    vals = []
    true_demand = []
    predicted_demand = []
    predicted_demand.append(0)
    lol = int(len(demand_data)/10)
    generate_model(demand_data.head(n=lol), './Models/model1')
    prev_pred = 0
    i = 0
    negative = 0 # How many predictions were underpredicted
    positive = 0 # How many of our predictions were overpredicted
    figure, axis = plt.subplots(2, 1)
    for i in range(lol, lol * 2):
        new_demand_data = demand_data.head(n=i)
        if (demand_data.iloc[-1]['value'] != -1):
            val = compute_prediction('./Models/model1', new_demand_data)
            vals.append(i)
            true_demand.append(val[0])
            dates.append(100 - (abs((prev_pred - val[0])) / val[0] * 100))
            if((abs((prev_pred - val[0])) / val[0]) < (1 - PREDICTION_THRESHOLD)):
                z = 0
                if((prev_pred - val[0]) < 0): negative += 1
                else: positive += 1
            prev_pred = val[1]
            predicted_demand.append(val[1])
    print("Negative: ", negative)
    print("Positive: ", positive)
    axis[0].plot(vals, dates)
    axis[0].set_ylim([-10, 110])
    axis[1].plot(vals, true_demand)
    vals.append(i)
    axis[1].plot(vals, predicted_demand)
    plt.savefig("matplotlib.png")
test_demonstration()
