# lstm for time series forecasting
from numpy import sqrt, asarray
import pandas as pd
from pandas import read_csv
from tensorflow import keras, math, reduce_mean
from keras import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
import time
from datetime import datetime
import sys
import random

N_STEPS = 5
SET_SIZE = 19
TEST_PROPORTION = 6

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

# Need new function to evaluate RMSE based on just 'Percent Output'
# def my_loss_fn(y_actual, y_pred):
#     SE = math_ops.squared_difference(y_pred.numpy()[0], y_actual.numpy()[0])  #squared difference
#     MSE = sqrt(SE)
#     return MSE

# a = keras.backend.constant([1,2,3])
# b = keras.backend.constant([4,5,6])

# loss = my_loss_fn(a,b)
# print(loss)

# load the dataset
df = read_csv("./Solar Forecasting/SolarTrainingData.csv", index_col=0)

df = df.drop(['year', 'day'], axis=1)
df.tail()
randstart = random.randint(0, df.size/SET_SIZE-N_STEPS)
df_predictor = asarray(df.iloc[randstart:randstart+N_STEPS+1]).reshape(1, N_STEPS+1, SET_SIZE)
dataframeset = [df.iloc[:randstart], df.iloc[randstart+N_STEPS+1:]]
df = pd.concat(dataframeset)

# retrieve the values
values = df.values.astype('float32')
# specify the window size
# split into samples
X, y = split_sequence(values, N_STEPS)
# reshape into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
# split into train/test
n_test = int(X.shape[0]/TEST_PROPORTION)
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# define model
# TODO: figure out how to set up the batchnormalization layer to be at the start, without normalizing percent output
model = Sequential()
model.add(BatchNormalization(input_shape=(N_STEPS+1,len(df.iloc[0]))))
# LSTM to weight recency
model.add(LSTM(100, activation='relu', kernel_initializer='he_normal'))
# dense layers to regenerate spatiality
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(25, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='tanh'))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# fit the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(X_test, y_test), callbacks=[callback])
# evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
# make a prediction
print(randstart)
actual_result = df_predictor[-1, -1, -1]
print('Real: %.3f' % (actual_result))
df_predictor[-1, -1, -1] = 0
print(df_predictor)
yhat = model.predict(df_predictor)
print('Predicted: %.3f' % (yhat))
acc = 100 - abs((actual_result - yhat) / actual_result * 100)
print('Accuracy: %.3f' % (acc))