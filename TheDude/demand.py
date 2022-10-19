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
VERBOSE = 0
PREDICTION_THRESHOLD = .96 # Percentage
DEMAND_UNINIT = 42069
# Dictionary key is cluster model path, value is list with [old prediction, counter]
cluster_predictions = {}

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


def sum_rows (values):
    difference = []
    difference.append(values[0])
    for i in range(1, len(values)):
        if values[i] - values[i - 1] < 0:
            difference.append(values[i - 1] - values[i - 2])
        else:
            difference.append(values[i] - values[i - 1])
    return difference

# This function fits the 'model' using an input pandas 'df' and the number of 'points' to fit on.
# 'points' is last number of points collected
# It also stores the fitted model in the filepath provided by 'model_location'
def fit_model(model, df, points, model_location):
    df = df.tail(n=points)
    values = df.loc[:,'value'].values
    values = (sum_rows(values))
    values[0] = 0
    # split into samples
    x, y = split_sequence(values)
    # reshape into [samples, timesteps, features]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    # split into train/test
    n_test = 12
    x_train, x_test, y_train, y_test = x[:-n_test], x[-n_test:], y[:-n_test], y[-n_test:]
    # fit the model
    little_x = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=VERBOSE, validation_data=(x_test, y_test))
    little_x.model.save(model_location)
    return x_test, y_test, little_x

# This function generates a demand model for LSTM given a dataframe with column parameter value
# that contains the data for amount of power used per time horizon. It assumes that the time
# between each data point is constant and returns the model.
def generate_model(df, model_location):
    # define model
    cluster_predictions[model_location] = [DEMAND_UNINIT, 0] #initialize all models [previous prediction, counter]
    if (not os.path.exists(model_location)):
        print(model_location)
        model = Sequential()
        model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(N_STEPS, 1)))
        model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(1))
        # compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        x_test, y_test, model = fit_model(model, df, len(df) if (NUM_DATA_POINTS == "MAX") else NUM_DATA_POINTS, model_location)

# This function generates a prediction given an input model and dataframe that contains the power consumption
# values in the value column. It will generate predictions for DEMAND_TIME_HORIZONS and update the model passed
# in with the new data point in the dataframe.
def compute_prediction(model_location, df):
    values = df.loc[:,'value'].values
    values = (sum_rows(values))
    current = values[len(values) - 1]
    th = []
    predict_model = keras.models.load_model(model_location)#this is copy that will be used to make predictions
    for i in range(settings.DEMAND_TIME_HORIZONS):
        row = asarray(values[-N_STEPS:]).reshape((1, N_STEPS, 1))
        th.append(predict_model.predict(row))
        values.append(th[i][0][0])
    accuracy = 1 #initialize accuracy to 1 in case 
    if (cluster_predictions[model_location][0] == DEMAND_UNINIT): cluster_predictions[model_location][0] = th[0][0][0]
    else: accuracy = abs((cluster_predictions[model_location][0] - current) / current)
    current_amount = wait_amount(model_location, False, True)
    # Update if batch size reached or predictions become inaccurate
    if (cluster_predictions[model_location][1] == (NEW_DATA_AMOUNT - 1)) or (accuracy < PREDICTION_THRESHOLD):
        fit_model(predict_model,df, current_amount)
        wait_amount(model_location, True, False) #reset counter if 

    return ([current] + [th[i][0][0] for i in range(settings.DEMAND_TIME_HORIZONS)])

#also look into retraining on whole data set every x amount of days

def accuracy(last_15min_predication, index):
    try:
        actual_result = sum(index)
    except:
        actual_result = index
    # actual_result = sum(values[index])
    acc = 100 - abs((actual_result - last_15min_predication) / actual_result * 100)
    return acc

def test_demonstration(model):
    predictions = []
    acc = []
    CSV = "BuildingData2018/ADH_E_TBU_CD_1514786400000_1535778000000_hourly.csv"
    true_data = [112, 109, 109]
    j = 0
    demand_data = pd.read_csv(CSV)
    dates = []
    vals = []
    lol = int(len(demand_data)/10)
    generate_model(demand_data.head(n=lol))
    prev_pred = 0
    i = 0
    # for i in range(lol, lol * 2):
    #     vals.append(i)
    #     new_demand_data = demand_data.head(n=i)
    #     val = compute_prediction('./Models/model1', new_demand_data, True, 50)
    #     dates.append(100 - abs((val[1][0] - prev_pred) / val[1][0] * 100))
    #     prev_pred = val[1][0]
    # plt.plot(vals, dates)
    # vals.append(i)
    # plt.plot(vals, TIME)
    # plt.ylim([80, 120])
    plt.show()
# tada = generate_demand_predictions("Demand Data/Annex West Active Power_August.csv")
# update = accuracy(100, "Demand Data/Running Data.csv")
# test_demonstration('hi')

