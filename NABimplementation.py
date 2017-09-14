
# coding: utf-8

# In[ ]:


import numpy as np
import SimAnomalyDataset as sim

# Get anomaly locations from simulation
data, anomaly_loc, anomaly_dur, dates = sim.get_data(n="n")

# Defined anomaly window size
N_anomalies = 10
windowSize = (len(data)*0.1)/N_anomalies # 10% of data is window size

# Sigmoid score function
def sigmoid(y, Atp= 1, Afp= -1):
    return((Atp-Afp)*(1/(1+np.exp(5*y)))-1)

# Raw score 
def raw_score(sig,Afn, fd):
    return np.sum(sig) + Afn*fd

# Sum raw scores
S = np.sum(raw_score(sigmoid))

# Normalization
Sperfect = 1
Snull = 0 
Snorm = 100*((S-Snull)/(Sperfect-Snull))

