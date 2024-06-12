# lstm for time series forecasting
from numpy import sqrt, asarray
from pandas import read_csv
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, BatchNormalization, LSTM, LayerNormalization
import matplotlib.pyplot as plt
from datetime import datetime

N_STEPS = 5
SET_SIZE = 19
PREDICTION_SET_SIZE = 580
TEST_PROPORTION = 25

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

# view normalized data

# load the dataset
df = read_csv("./Solar Forecasting/SolarTrainingData.csv", index_col=0)

df_normalized = df.copy()

column_mean = []
column_std = []

# apply normalization techniques
for column in df_normalized.columns:
    if column != 'Percent Output':
        column_mean.append(df_normalized[column].mean())
        column_std.append(df_normalized[column].std())

        if df_normalized[column].std() == 0:
            df_normalized.loc[:,column] = 0
        else:
            df_normalized[column] = (df_normalized[column] - df_normalized[column].mean()) / df_normalized[column].std()


df_normalized = df_normalized.drop(['feelslike', 'year', 'day'], axis=1)
df_holdout = df_normalized.iloc[-PREDICTION_SET_SIZE:]
df_normalized = df_normalized.iloc[:-PREDICTION_SET_SIZE]

# retrieve the values
values = df_normalized.values.astype('float32')
# specify the window size
# split into samples
X, y = split_sequence(values, N_STEPS)
# split into train/test
n_test = int(X.shape[0]/TEST_PROPORTION)
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# define model
# TODO: figure out how to set up the batchnormalization layer to be at the start, without normalizing percent output
model = Sequential()
#model.add(BatchNormalization(input_shape=(N_STEPS+1,len(df_normalized.iloc[0]))))
# LSTM to weight recency
model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', return_sequences=True, input_shape=(N_STEPS+1,len(df_normalized.iloc[0]))))
model.add(LayerNormalization())
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LayerNormalization())
model.add(LSTM(25, activation='relu'))
model.add(LayerNormalization())
model.add(Dense(1, activation='tanh'))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# fit the model
model.fit(X_train, y_train, epochs=7, batch_size=20, verbose=2, validation_data=(X_test, y_test), callbacks=[callback])
# evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
# make a prediction

holdout_X, holdout_y = split_sequence(df_holdout.values.astype('float32'), N_STEPS)
yhat = model.predict(holdout_X)
print(yhat)
figure, axis = plt.subplots(1, 1)
axis.plot([i for i in range(len(holdout_y))], holdout_y)
axis.plot([i for i in range(len(yhat))], yhat)
plt.savefig("Solar Forecasting/matplotlib.png")
#for i in range(len(holdout_X)):
#    print('Real: %.3f' % (holdout_y[i]))
#    prediction_set = holdout_X[i].reshape(1, holdout_X[i].shape[0], holdout_X[i].shape[1])
#    yhat = model.predict(prediction_set)
#    print('Predicted: %.3f' % (yhat))
#    acc =  abs((holdout_y[i] - yhat) / holdout_y[i] * 100)
#    print('Accuracy: %.3f\n' % (acc))
