import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn import *
from datetime import datetime

pd.read_csv('CSV Data/Annex East Active Power_August.csv')

AnnexEastAug = pd.read_csv('CSV Data/Annex East Active Power_August.csv')
AnnexEastFeb = pd.read_csv('CSV Data/Annex East Active Power_Feb.csv')

HarrisSouthAug = pd.read_csv('CSV Data/Harris North South Active Power_August.csv')
HarrisSouthFeb = pd.read_csv('CSV Data/Harris North South Active Power_Feb.csv')

#Display CSV as an example
pd.read_csv('CSV Data/Annex East Active Power_August.csv')


# Return Mondays, can be quickly mofified to return any day of the week
def filt(row: pd.Series) -> bool:
    return pd.Timestamp(
        datetime.strptime(row['DateTime'].split()[0], '%m/%d/%Y').strftime('%Y-%m-%d')).day_name() == 'Monday'


AnnexEastAugMonday = AnnexEastAug[AnnexEastAug.apply(filt, axis=1)]
print(AnnexEastAugMonday)

#Describe numerical statistics of data including things like mean, std dev, etc
AnnexEastAug.describe()

#Describe numerical statistics of data including things like mean, std dev, etc
AnnexEastFeb.describe()

#Describe numerical statistics of data including things like mean, std dev, etc
HarrisSouthAug.describe()

#Describe numerical statistics of data including things like mean, std dev, etc
HarrisSouthFeb.describe()

# #Plot histograms-displays a histogram for each buildings(or sections) across all demands
# AnnexEastAug.select_dtypes(include=np.number).hist(figsize=(80,40))
# plt.tight_layout()
# plt.show()
#
# #Plot histograms-displays a histogram for each buildings(or sections) across all demands
# AnnexEastFeb.select_dtypes(include=np.number).hist(figsize=(80,40))
# plt.tight_layout()
# plt.show()
#
# #Plot histograms-displays a histogram for each buildings(or sections) across all demands
# HarrisSouthAug.select_dtypes(include=np.number).hist(figsize=(80,40))
# plt.tight_layout()
# plt.show()
#
# #Plot histograms-displays a histogram for each buildings(or sections) across all demands
# HarrisSouthFeb.select_dtypes(include=np.number).hist(figsize=(80,40))
# plt.tight_layout()
# plt.show()
#
# #Create correlation matrix
# correlationMatrix = AnnexEastAug.corr()
# plt.figure(figsize=(15,10))
# sn.heatmap(correlationMatrix, annot=False)
# plt.show()
#
# #Create correlation matrix
# correlationMatrix = AnnexEastFeb.corr()
# plt.figure(figsize=(15,10))
# sn.heatmap(correlationMatrix, annot=False)
# plt.show()
#
# #Create correlation matrix
# correlationMatrix = HarrisSouthAug.corr()
# plt.figure(figsize=(15,10))
# sn.heatmap(correlationMatrix, annot=False)
# plt.show()
#
# #Create correlation matrix
# correlationMatrix = HarrisSouthFeb.corr()
# plt.figure(figsize=(15,10))
# sn.heatmap(correlationMatrix, annot=False)
# plt.show()

import tensorflow
from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import jinja2

# recurrent neural network
# natural langauge processing problems
# text provided as input to the model
# Long Term Short Term network
# assign class or predict numerical value
# split into time steps and observations

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

def get_date(dates):
    for date in dates:
        print(date)
        yield date

def sum_rows(dates, values):
    print("hi")
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
    dataframe = pd.DataFrame(csv, columns =['DateTime', 'Summed Power'])
    dataframe.style.hide_index()
    dataframe.to_csv("CSV Data/Running Data.csv")

# load the dataset
CSV = 'CSV Data/Annex East Active Power_Feb.csv'
print("Recurrent Neural Network")
path = CSV
df = read_csv(path, header=0, index_col=0, squeeze=True)
# retrieve the values
dates = []
import csv
with open(CSV, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        if row[0] != 'DateTime':
            dates.append(row[0])
print(dates)
values = df.values.astype('float32')
sum_rows(dates, values)
df = read_csv("CSV Data/Running Data.csv", header=0, index_col=0, squeeze=True)
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
# make a prediction
row = asarray([0.335600019,0.316700041, 0.336100012,0.280300021, 0.282499999]).reshape((1, n_steps, 1))
yhat = model.predict(row)
print('Predicted: %.3f' % (yhat))
