
# coding: utf-8

# In[ ]:


import numpy as np
import scipy

"""
Functions for computing whether or not predictions are anomalies or not
"""

def prediction_error(predicted, expected):
        """
        calculate prediction error
        """
        T = np.size(predicted)
        S = np.zeros(T)

        # Loop across rows
        for t in np.arange(T):
            # Commonality between matricies
            top = np.dot(predicted[t],expected[t]) 

            # Total number of bits in ISM
            bottom = np.sum(expected[t]) # not norm*

            # Calculate Prediction error
            S[t] = 1 - (top/(bottom))
        
        return S

def anomaly_likelihood(S):
    """
    Calculate the anomaly likelihood

    """ 
    T = np.size(S)
    L = np.zeros(T)

    # Model distribution of Prediction error values as "rolling normal distribution"
    W = 10 # Long-term window size
    w = 2  # Short-term window size
    epsilon = 10**-5 # Anomaly threshold

    for t in np.arange(T):
        W_range = np.arange(t,t-(W-1),-1)
        w_range = np.arange(t,t-(w-1),-1)

        # Long-term window mean
        mew_W = np.sum(S[W_range])/ W

        # Normal window standard deviation
        sigma_W = np.sum((S[W_range]-mew_W)**2) / (W-1)

        # Short-term window mean
        mew_w = np.sum(S[w_range])/ w 

        # Anomaly likelihood
        L[t] = 1 - Q_func((mew_w - mew_W)/sigma_W)

        # Threshold where likelihood very high [> 0.99999]
        Anomaly = L[L>=1-epsilon]   

    return L
    

def Q_func(x):
    """
    Normal distribution tail function approximation
    """
    return (1/2)*scipy.special.erf(x/np.sqrt(2))
    

#S = prediction_error(predicted=predictions, expected=raw_values)
#A = anomaly_likelihood(S)

