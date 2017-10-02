
# coding: utf-8

# In[1]:


import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt
import SimAnomalyDataset as sim

"""
Main code of LSTM anomaly detection, including data preparation

"""

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# create a differenced series
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# make a persistence forecast
def persistence(last_ob, n_seq):
    return [last_ob for i in np.arange(n_seq)]
 
# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
    forecasts = []
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = persistence(X[-1], n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = []
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test,var):
    inverted = []
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series[var][index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[(n_lag+i)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))

# Create dataset
data, anomaly_loc, anomaly_dur, dates = sim.get_data(n=0,datalabels=["timestamp","consumption"])
        
# Import dataset
series = read_csv('sim_data.csv', header=0, index_col=0, squeeze=True)
#series = read_csv('all_month.csv', header=0, index_col=0, squeeze=True)
#series = series.filter(['time','mag'], axis=1)
#series = series[np.isfinite(series['mag'])]

var = "consumption"

# configure
n_lag = 1
n_seq = 1
n_test = 2000
n_epochs = 1500
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test =  prepare_data(series[var], n_test, n_lag, n_seq)
# make forecasts
forecasts = make_forecasts(train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2,var)


# In[2]:


import SimAnomalyDataset as sim
import evaluatePredictions as evalPred
data, anomaly_loc, anomaly_dur, dates = sim.get_data()

"""
Evaluate predictions based on various statistical metrics - generates prediction error scaled between 0-1
"""

actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2,var)

actual2=np.reshape(actual,(2000,1))
forecasts2=np.reshape(forecasts,(2000,1))

# Compute comparison metric for predicted vs input (anomalies) [PICK CHANNEL]
metric = "RMSE"
out = evalPred.main(actual2[:,0], forecasts2[:,0], metric=metric)

#Threshold metric before plotting
thresh_min = 0
thresh_max = 10000
to_plot = out
to_plot=np.asarray(to_plot)
to_plot[to_plot<thresh_min]=0
to_plot[to_plot>thresh_max]=thresh_max
to_plot = to_plot/np.max(to_plot)

sim.plot_data(to_plot, anomaly_loc, anomaly_dur,title=metric)


# In[3]:


# Fit prediction error with Gaussian to extract anomaly score
pred = actual2[:,0]- forecasts2[:,0]
anomaly_score = evalPred.GaussianPredError(pred, anomaly_loc, anomaly_dur,thresh=0.05)

# sim.plot_data(anomaly_score, anomaly_loc, anomaly_dur,title=metric)
# plt.show()


# In[4]:


"""
Write to NUMENTA style csv file (Also for NAB)
"""

# Convert anomaly locations to binary array
labels = np.zeros_like(dates)
for i in anomaly_loc:
    labels[i] = 1
    
# Pad anomaly scores to zero during training
full_anomaly_scores = np.zeros_like(dates)
full_anomaly_scores[8000:] = anomaly_score

# Write data to CSV file
datacsv = DataFrame()
datacsv["timestamp"] = dates
datacsv["value"] = np.round(data,3)
datacsv["anomaly_score"] = full_anomaly_scores
datacsv["label"] = labels

## Write CSV to folder
#datacsv.to_csv(path_or_buf="/home/codepan1/RestRunnerCode/alpha_Twitter_volume_AAPL.csv")
#series = read_csv("/home/codepan1/RestRunnerCode/alpha_Twitter_volume_AAPL.csv")



# In[5]:


import NABimplementation as NAB

labels = datacsv["label"].values
anomaly_score = datacsv["anomaly_score"].values
NAB.main(labels, anomaly_score)


# In[40]:


# Save NAB for various noise levels

NAB_noise = []
for n in np.linspace(0,0.5,10): # percentage of base signal amplitude
    # create dataset
    data, anomaly_loc, anomaly_dur, dates = sim.get_data(n,datalabels=["timestamp","consumption"])
    
    # read csv
    series = read_csv('sim_data.csv', header=0, index_col=0, squeeze=True)
    
    # run LSTM
    var = "consumption"
    n_lag = 1
    n_seq = 1
    n_test = 2000
    n_epochs = 1500
    n_batch = 1
    n_neurons = 1
    # prepare data
    scaler, train, test =  prepare_data(series[var], n_test, n_lag, n_seq)
    # make forecasts
    forecasts = make_forecasts(train, test, n_lag, n_seq)
    # inverse transform forecasts and test
    forecasts = inverse_transform(series, forecasts, scaler, n_test+2,var)
   
    # reshape predictions and input
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(series, actual, scaler, n_test+2,var)
    actual2=np.reshape(actual,(2000,1))
    forecasts2=np.reshape(forecasts,(2000,1))
    
    # Fit prediction error with Gaussian to extract anomaly score
    pred = actual2[:,0]- forecasts2[:,0]
    anomaly_score = evalPred.GaussianPredError(pred, anomaly_loc, anomaly_dur,thresh=0.05)
    
    # Convert anomaly locations to binary array
    labels = np.zeros_like(dates)
    for i in anomaly_loc:
        labels[i] = 1
        
    # Pad anomaly scores to zero during training
    full_anomaly_scores = np.zeros_like(dates)
    full_anomaly_scores[8000:] = anomaly_score
    
    # calculate NAB
    NAB_noise.append(NAB.main(labels, full_anomaly_scores))


# In[44]:


x = np.linspace(0,0.5,10)*100
plt.plot(x, NAB_noise,'x')
plt.xlabel("Percentage noise")
plt.ylabel("NAB score")
#plt.xlim([0,25])
#plt.ylim([-1,100])
plt.show()


# In[34]:




