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
import csv
import jinja2

amount_of_data_analyzed = 300

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
    difference = []
    summed_data.append(dates[0])
    summed_data.append(values[0])
    difference.append(values[0])
    csv.append(summed_data)
    for i in range(1, len(values)):
        summed_data = []
        summed_data.append(dates[i])
        if values[i] - values[i - 1] < 0:
            summed_data.append(values[i - 1] - values[i - 2])
            difference.append(values[i - 1] - values[i - 2])
        else:
            summed_data.append(values[i] - values[i - 1])
            difference.append(values[i] - values[i - 1])
        csv.append(summed_data)
    dataframe = pd.DataFrame(csv, columns=['DateTime', 'value'])
    dataframe.style.hide_index()
    dataframe.to_csv("Demand Data/Running Data.csv", index=False)
    return difference


def machine_learning(df, dates):
    values = df.loc[:,'value'].values
    values = (sum_rows(dates, values))[len(values) - amount_of_data_analyzed: len(values) - 1]
    n_steps = 5
    # split into samples
    X, y = split_sequence(values, n_steps)
    # reshape into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # split into train/test
    n_test = 12
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_steps, 1)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(X_test, y_test))
    # evaluate the model
    mse, mae = model.evaluate(X_test, y_test, verbose=0)
    # print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
    # make a prediction for 15, 30, 45, and 60 minutes
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
    # print('Predicted: %.3f' % (yhat))
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
            if row[0] != 'human_timestamp':
                dates.append(row[0])
    return machine_learning(df, dates)

def accuracy(last_15min_predication, CSV):
    df = read_csv(CSV, header=0, index_col=0, squeeze=True)
    values = df.values.astype('float32')
    values = values.tolist()
    try:
        actual_result = sum(values[len(values) - 1])
    except:
        actual_result = values[len(values) - 1]
    # actual_result = sum(values[index])
    acc = 100 - abs((actual_result - last_15min_predication) / actual_result * 100)
    return acc

def test_demonstration():
    predictions = []
    acc = []
    CSV = "BuildingData2018/ADH_E_TBU_CD_1514786400000_1535778000000_hourly.csv"
    true_data = [664.6799945831299, 673.8599910736084, 644.7099781036377, 638.129976272583]
    demand_data = pd.read_csv(CSV)
    dates = []
    for i in range(demand_data.size - 4, demand_data.size):
        dates = []
        with open(CSV, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if row[0] != 'timestamp':
                    dates.append(row[0])
        current_predictions = machine_learning(demand_data[0:i], dates[0:i])
        predictions.append(current_predictions)
        try:
            if len(predictions) > 1:
                latest = (predictions[len(predictions) - 2][1])
                print(latest)
                update = accuracy(latest, "Demand Data/Running Data.csv")
                print(update)
                acc.append(update)
        except:
            nothing = 0
    print("Predictions")
    print(predictions)
    print("Accuracy after each prediction")
    print(acc)
    # tada = generate_demand_predictions("Demand Data/Annex West Active Power_August.csv")
    # print(tada)
# update = accuracy(100, "Demand Data/Running Data.csv")
# test_demonstration()

