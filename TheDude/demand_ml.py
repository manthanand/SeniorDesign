import pandas as pd
from datetime import datetime
import math
import tensorflow
from numpy import concatenate
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import jinja2
import csv


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)


def sum_rows(dates, values):
    csv = []
    summed_data = []
    for value, date in zip(values, dates):
        total = 0
        for num in value:
            total += num
        summed_data.append(date)
        summed_data.append(total)
        csv.append(summed_data)
        summed_data = []
    dataframe = pd.DataFrame(csv, columns=['DateTime', 'Summed Power'])
    dataframe.style.hide_index()
    dataframe.to_csv("CSV Data/Running Data.csv", index=False)


def machine_learning(df, dates):
    values = df.values.astype('float32')
    sum_rows(dates, values)
    df = read_csv("CSV Data/Running Data.csv", header=0, index_col=0, squeeze=True)
    values = df.values.astype('float32')
    # specify the window size
    n_steps = 5
    # split into samples
    X, y = split_sequence(values, n_steps)
    # reshape into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # split into train/test
    n_test = 12
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps, 1)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # fit the model
    model.fit(X_train, y_train, epochs=350, batch_size=32, verbose=2, validation_data=(X_test, y_test))
    # evaluate the model
    mse, mae = model.evaluate(X_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
    # make a prediction for 15, 30, 45, and 60 minutes
    values = values.tolist()
    current = values[len(values) - 1]
    yhat_15min = compute_prediction(model, values[-5:], n_steps)
    values.append(yhat_15min[0][0])
    yhat_30min = compute_prediction(model, values[-5:], n_steps)
    values.append(yhat_30min[0][0])
    yhat_45min = compute_prediction(model, values[-5:], n_steps)
    yhat_45min = yhat_45min.tolist()
    values.append(yhat_45min[0][0])
    yhat_60min = compute_prediction(model, values[-5:], n_steps)
    return [current, yhat_15min[0][0], yhat_30min[0][0], yhat_45min[0][0], yhat_60min[0][0]]


# Commented out since we won't know the true value when we make our predication
# def compute_prediction(model, five_entries, actual_result, n_steps):
# NOTE: need to optimize this somehow
# row = asarray(
# five_entries).reshape(
# (1, n_steps, 1))
# yhat = model.predict(row)
# print('Predicted: %.3f' % (yhat))
# accuracy = 100 - abs((actual_result - yhat) / yhat) * 100
# print('Accuracy: ', accuracy)


def compute_prediction(model, five_entries, n_steps):
    # NOTE: need to optimize this somehow
    row = asarray(
        five_entries).reshape(
        (1, n_steps, 1))
    yhat = model.predict(row)
    print('Predicted: %.3f' % (yhat))
    return yhat


def generate_demand_predictions(CSV):
    # load the dataset
    path = CSV
    df = read_csv(path, header=0, index_col=0, squeeze=True)
    # retrieve the values
    dates = []
    with open(CSV, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] != 'DateTime':
                dates.append(row[0])
    return machine_learning(df, dates)


def main():
    tada = generate_demand_predictions("../CSV Data/Annex West Active Power_August.csv")
    print(tada)
