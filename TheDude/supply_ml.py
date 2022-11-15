import keras
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import math
from tensorflow import reduce_mean
import tensorflow as tf
import settings

N_STEPS = 5
NUM_EPOCHS = 100
NEW_DATA_AMOUNT = 168
VERBOSE = 2
PREDICTION_SET_SIZE = 580
N_TEST = 50

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
    print('X: \n', X[0:5], '\n Y: \n', y[0:5])
    return asarray(X), asarray(y)

def custom_loss(y_actual, y_pred):
    MSE = reduce_mean(math.square(y_pred - y_actual), axis=0)
    return tf.where([False for i in range(20)] + [True], MSE, tf.zeros_like(MSE))  # create tensor where every loss is 0 except solar output

def custom_eval(y_actual, y_pred):
    MSE = reduce_mean(math.square(y_pred - y_actual), axis=0)
    return tf.math.sqrt(tf.where([False for i in range(20)] + [True], MSE, tf.zeros_like(MSE)))


def generate_model(starting, model_location, csv):
    df = read_csv(csv, index_col=0)
    df = df.head(n=starting)
    df_holdout = df.iloc[-PREDICTION_SET_SIZE:]
    df = df.iloc[:-PREDICTION_SET_SIZE]

    holdout_X, holdout_y = split_sequence(df_holdout.values.astype('float32'), N_STEPS)
    # reshape into [samples, timesteps, features]
    holdout_X = holdout_X.reshape((holdout_X.shape[0], holdout_X.shape[1], 21))
    # split into train/test
    print(holdout_X[:-N_TEST])
    X_train, X_test, y_train, y_test = holdout_X[:-N_TEST], holdout_X[-N_TEST:], holdout_y[:-N_TEST], holdout_y[-N_TEST:]
    # improvement area : try adding dropout
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(N_STEPS, 21)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    # fit the model
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=2, validation_data=(X_test, y_test))
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
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    little_x = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=VERBOSE, validation_data=(x_test, y_test))
    little_x.model.save(model_location)
    return x_test, y_test, little_x


def compute_prediction(model_location, df):
    current = df.tail(1)
    operate_current = current
    th = []
    predict_model = keras.models.load_model(model_location, 
        custom_objects={"custom_eval": custom_eval, "custom_loss": custom_loss}, 
        compile=False)#this is copy that will be used to make predictions
    for i in range(settings.SUPPLY_TIME_HORIZONS):
        operate_current, y = split_sequence(df[-N_STEPS - 1:].values.astype('float32'), N_STEPS)
        operate_current = operate_current.reshape((operate_current.shape[0], operate_current.shape[1], 21))
        prediction = predict_model.predict(operate_current, verbose=VERBOSE)
        th.append(prediction)
        df.append(prediction)
    return prediction
