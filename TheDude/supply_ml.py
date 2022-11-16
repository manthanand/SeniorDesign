import keras
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow import math
from tensorflow import reduce_mean
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
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
        if end_ix > len(sequence)-1: break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

def custom_loss(y_actual, y_pred):
    MSE = reduce_mean(math.square(y_pred - y_actual), axis=0)
    return tf.where([False for i in range(16)] + [True], MSE, tf.zeros_like(MSE))  # create tensor where every loss is 0 except solar output


def custom_eval(y_actual, y_pred):
    MSE = reduce_mean(math.square(y_pred - y_actual), axis=0)
    return tf.math.sqrt(tf.where([False for i in range(16)] + [True], MSE, tf.zeros_like(MSE)))

def generate_model(starting, model_location, csv):
    ''' This is MLP version that was less accurate than LSTM
    df = pd.read_csv(csv, index_col=0)
    df = df.head(n=starting)
    column_mean = []
    column_std = []
    pastWeatherAndSupply = df.iloc[:-2]

    currentWeather = df.iloc[-2]
    currentWeather = currentWeather.drop(['Percent Output'])  # dropping supply column for testing
    currentWeather = currentWeather.values[:].reshape((1, len(df.iloc[-2]) - 1))

    nextHourWeather = df.iloc[-1]
    nextHourWeather = nextHourWeather.drop(['Percent Output'])  # dropping supply column for testing
    nextHourWeather = nextHourWeather.values[:].reshape((1, len(df.iloc[-1]) - 1))
    X, y = pastWeatherAndSupply.values[:, :-1], pastWeatherAndSupply.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    n_features = X_train.shape[1]
    print(n_features)
    model = Sequential()
    model.add(BatchNormalization(input_shape=(n_features,)))
    model.add(Dense(12, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(8, activation='selu', kernel_initializer='he_normal'))
    model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='tanh', kernel_initializer='he_normal'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=80, batch_size=10)
    model.save(model_location)'''

    df = read_csv(csv, index_col=0)
    df = df.head(n=starting)
    # retrieve the values
    values = df.values.astype('float32')
    # split into samples
    X, y = split_sequence(values, N_STEPS)
    # split into train/test
    X_train, X_test, y_train, y_test = X[:-N_TEST], X[-N_TEST:], y[:-N_TEST], y[-N_TEST:]
    # define model
    # improvement area : try adding dropout
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='he_normal', input_shape=(N_STEPS, 17)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    # compile the model
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    # fit the model
    model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=2, validation_data=(X_test, y_test), callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3))
    model.save(model_location)

def fit_model(model, df, points, model_location, n_tests):
    df = df.tail(n=points)
    values = df.loc[:,'value'].values
    # split into samples
    X, y = split_sequence(values)
    # reshape into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 17))
    # split into train/test
    x_train, x_test, y_train, y_test = X[:-n_tests], X[-n_tests:], y[:-n_tests], y[-n_tests:]
    # fit the model
    model.compile(optimizer='adam', loss=custom_loss, metrics=[custom_eval])
    little_x = model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=32, verbose=VERBOSE, validation_data=(x_test, y_test))
    little_x.model.save(model_location)
    return x_test, y_test, little_x


def compute_prediction(model_location, df):
    current = df['Percent Output'].values[-1]
    operate_current = current
    predict_model = keras.models.load_model(model_location, custom_objects={"custom_eval": custom_eval, "custom_loss": custom_loss})
    operate_current, y = split_sequence(df[-N_STEPS - 1:].values.astype('float32'), N_STEPS)
    prediction = predict_model.predict(operate_current, verbose=VERBOSE)
    LUT = pd.read_csv(settings.lookuptable)['P Output [MW]']
    if current <= 0:
        return [0, 0]
    elif (prediction <= 0):
        return [LUT[abs(current*1000).round()]*1000, 0]
    return [LUT[abs(current*1000).round()]*1000, LUT[abs(prediction[0][0]*1000).round()]*4]

'''    currentWeather = df.iloc[-2]
    currentWeather = currentWeather.drop(['Percent Output'])  # dropping supply column for testing
    currentWeather = currentWeather.values[:].reshape((1, len(df.iloc[-2]) - 1))

    nextHourWeather = df.iloc[-1]
    nextHourWeather = nextHourWeather.drop(['Percent Output'])  # dropping supply column for testing
    nextHourWeather = nextHourWeather.values[:].reshape((1, len(df.iloc[-1]) - 1))
    model = keras.models.load_model(model_location)#this is copy that will be used to make predictions
    current_prediction = model.predict(currentWeather)
    if (current_prediction < 0):
        current_prediction = 0
    print('Predicted: %.3f' % current_prediction)
    print('Real: %.3f' % df.iloc[-2][-1])
    print(abs(df.iloc[-2][-1] - current_prediction) / df.iloc[-2][-1])

    future_prediction = model.predict(nextHourWeather)
    if (future_prediction < 0):
        future_prediction = 0
    print('Predicted: %.3f' % future_prediction)
    print('Real: %.3f' % df.iloc[-1][-1])
    print(abs(df.iloc[-1][-1] - future_prediction) / df.iloc[-1][-1])'''

# compute_prediction(settings.modelfp + "Supply", read_csv((glob.glob(settings.supplyfp)[0]), header=0, index_col=0).head(100))


