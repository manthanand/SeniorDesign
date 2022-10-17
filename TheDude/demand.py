import pandas as pd
from datetime import datetime
import math
#import tensorflow
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


DATA_POINTS = 300
VAR = 300
NUM_EPOCHS = 100
N_STEPS = 5
NEW_DATA_AMOUNT = 168
cluster_images = []

#waits for retraining
def wait_amount():
    current = 0
    if current == NEW_DATA_AMOUNT:
        current = 0
    else:
        current += 1
    yield current

# split a univariate sequence into samples
def split_sequence(sequence):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + N_STEPS
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)


def sum_rows (values):
    difference = []
    difference.append(values[0])
    for i in range(1, len(values)):
        if values[i] - values[i - 1] < 0:
            difference.append(values[i - 1] - values[i - 2])
        else:
            difference.append(values[i] - values[i - 1])
    return difference

def fit_model(model, df, points, model_location):
    df = df.tail(n=points)
    values = df.loc[:,'value'].values
    values = (sum_rows(values))
    values[0] = 0
    # split into samples
    X, y = split_sequence(values, N_STEPS)
    # reshape into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # split into train/test
    n_test = 12
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
    # fit the model
    little_x = model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=2, validation_data=(X_test, y_test))
    little_x.model.save(model_location)
    return X_test, y_test, little_x

# This function generates a demand model for LSTM given a dataframe with column parameter value
# that contains the data for amount of power used per time horizon. It assumes that the time
# between each data point is constant and returns the model.
def generate_model(df, model_location):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(N_STEPS, 1)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    X_test, y_test, model = fit_model(model, df, len(df))
    model.model.save(model_location)
    model = keras.models.load_model(model_location)
    # return model


# This function generates a prediction given an input model and dataframe that contains the power consumption
# values in the value column. It will generate predictions for DEMAND_TIME_HORIZONS and update the model passed
# in with the new data point in the dataframe. If you do not wish the update the original model, set 
# update = False. This function returns a tuple (updatedmodel - Sequential() type, predictions - list type)
def compute_prediction(model, df, update, points):
    # NOTE: need to optimize this somehow
    values = df.loc[:,'value'].values
    values = (sum_rows(values))
    current = values[len(values) - 1]
    th = []
    predict_model = keras.models.load_model(model)#this is copy that will be used to make predictions
    for i in range(settings.DEMAND_TIME_HORIZONS):
        row = asarray(values[-N_STEPS:]).reshape((1, N_STEPS, 1))
        th.append(predict_model.predict(row))
        values.append(th[i][0][0])
    update = (wait_amount == NEW_DATA_AMOUNT)
    if update:
        model = fit_model(predict_model,df, points)
        # df = df.
        # generate_model(df)
        # return machine_learning(df)
    return (model, [current] + [th[i][0][0] for i in range(settings.DEMAND_TIME_HORIZONS)])

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

