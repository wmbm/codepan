
# coding: utf-8

# In[1]:


#from sklearn.metrics import mean_squared_error
import math 
import scipy.stats as stats
import numpy as np

def main(test, pred, metric = "RMSE"):
    """
    Various metric for comparing output predictions and input values for 
    learning algorithms
    
    RSME, MAPE, MPE, MAD, NLL, MSE
    """
    
    N = len(pred)
    
    if metric == "RMSE":
        out = []
        for i in np.arange(N):
            out.append(math.sqrt(((test[i] - pred[i])**2)/2))
    
    elif metric == "MAPE":
        out = []
        for i in np.arange(N):
            out.append(((np.abs((test[i] - pred[i])/test[i]))/2)*100)
        
    elif metric == "MPE": 
        out = []
        for i in np.arange(N):
            out.append((((test[i] - pred[i])/pred[i])/2))
        
    elif metric == "MAD":
        out = []
        for i in np.arange(N):
            out.append(np.abs(test[i] - pred[i])/2)
    
    elif metric == "NLL":
        out = []
        for i in np.arange(N):
            out.append(-np.sum(stats.norm.logpdf(test[i], loc=pred[i]))/100)
    
    elif metric == "MSE":
        out = []
        for i in np.arange(N):
            out.append(((test[i] - pred[i])**2)/2)
    
    else:
        print("Key error")
    
    return out

