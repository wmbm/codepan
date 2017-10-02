
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
import evaluatePredictionError as evalPred

"""
Main code of LSTM anomaly detection, including data preparation

"""

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values)
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
def inverse_transform(series, forecasts, scaler, n_test):
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
        last_ob = series.values[index]
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

def main(raw_data,n_lag = 1,n_seq = 3,n_test = 2000,n_epochs = 1500, n_batch = 1, n_neurons = 1,sim="n"):
    """
    LSTM-AD - Long-Short term memory anomaly detection algorithm
    
    Parameters
    ----------
    n_lag :   comparison step into the past
    n_seq : prediction step into the future
    n_test : length of test series
    n_epochs : number of epochs
    n_batch : batch size (1 for time series datasets)
    n_neurons : number of neurons in networks
    sim : is this being run on simulated dataset with known anomalies?
    
    Returns
    ----------
    """
    # prepare data
    scaler, train, test =  prepare_data(raw_data, n_test, n_lag, n_seq)
    print("data prepared....")
    # make forecasts
    forecasts = make_forecasts(train, test, n_lag, n_seq)
    print("forecasts made....")
    # inverse transform forecasts and test
    forecasts = inverse_transform(raw_data, forecasts, scaler, n_test+2)
    print("forecasts inversed....")
        
    #Evaluate predictions - scaled between 0-1
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(raw_data, actual, scaler, n_test+2)

    actual2=np.reshape(actual,(n_test,n_seq))
    forecasts2=np.reshape(forecasts,(n_test,n_seq))

    # Compute comparison metric for predicted vs input (anomalies) [PICK CHANNEL] ######METHOD 1
    metric = "RMSE"
    out = evalPred.main(actual2[:,0], forecasts2[:,0], metric=metric)
    print("predictions evaluated....")

    # Fit prediction error with Gaussian to extract anomaly score ####### METHOD 2
    #pred = actual2[:,0]- forecasts2[:,0]
    #anomaly_score = evalPred.GaussianPredError(pred ,thresh=0.05)  
    #print("anomaly score generated....")
    
    return out, forecasts2


# In[ ]:


#     #Threshold metric before plotting
#     thresh_min = 0
#     thresh_max = 10000
#     to_plot = out
#     to_plot=np.asarray(to_plot)
#     to_plot[to_plot<thresh_min]=0
#     to_plot[to_plot>thresh_max]=thresh_max
#     to_plot = to_plot/np.max(to_plot)


# In[40]:


# Save NAB for various noise levels

def LSTM_noise(max_noise=0.5, steps = 10):
    """
    Model LSTM with varying simulated noise levels and plot output of detector score against time
    
    """
    NAB_noise = []
    for n in np.linspace(0,max_noise,steps): # percentage of base signal amplitude
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
        labels = np.zeros_like(data)
        for i in anomaly_loc:
            labels[i] = 1

        # Pad anomaly scores to zero during training
        full_anomaly_scores = np.zeros_like(data)
        full_anomaly_scores[8000:] = anomaly_score

        # calculate NAB
        NAB_noise.append(NAB.main(labels, full_anomaly_scores))
    
    x = np.linspace(0,0.5,10)*100
    plt.plot(x, NAB_noise,'x')
    plt.xlabel("Percentage noise")
    plt.ylabel("NAB score")
    #plt.xlim([0,25])
    #plt.ylim([-1,100])
    plt.show()

