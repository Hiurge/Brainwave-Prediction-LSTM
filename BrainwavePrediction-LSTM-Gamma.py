import numpy as np
from numpy import concatenate
from math import sqrt
import scipy
import scipy.stats as stats
import statsmodels
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt 
from matplotlib import pyplot
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import os

# DATA 
muse_df = pd.read_csv('Muse_big.csv')
del muse_df["Battery"]
del muse_df["Elements"]
del muse_df["HeadBandOn"]

# DATA CLEANING
muse_df = muse_df.iloc[161:, 1:21] # dropping TimeStamp, no-needed columns, few initial rows
muse_df = muse_df.dropna()
muse_df.reset_index(level=None, drop=True, inplace=True, col_level=0, col_fill='') 
muse_df.index += 1 
values = muse_df.values
values = values.astype('float32')


#
#
#
#
#
#
#
#
#

# DATA PREP FOR SUPERVISED LEARNING
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
reframed = series_to_supervised(scaled, 1, 1)

columns_to_stay = []
for i in range(20,40):
    if i not in [36,37,38,39]: 
        columns_to_stay.append(i)
reframed.drop(reframed.columns[columns_to_stay], axis=1, inplace=True)

values = reframed.values


# TRAIN/TEST SPLIT
train_break = 80000
test_break = 180000
train = values[:train_break, :]
test = values[train_break:test_break, :]

train_X, train_y = train[:, :-4], train[:, -1:]
test_X, test_y = test[:, :-4], test[:, -1:]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# FIT DATA INTO MODEL
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

my_epochs = 50
my_batch_size= 16 * 16 * 2

history = model.fit(train_X, train_y, epochs=my_epochs, batch_size=my_batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# PREDICTION
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print("Train/Test size:",train_break,"/",test_break-train_break,"Epochs:",my_epochs, "Batch-size:",my_batch_size,"#Features:", train_X.shape[2])
model.summary()