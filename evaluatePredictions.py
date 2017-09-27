
# coding: utf-8

# In[1]:


#from sklearn.metrics import mean_squared_error
import math 
import scipy.stats as stats
import numpy as np
import SimAnomalyDataset as sim
import matplotlib.pylab as plt

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


def GaussianPredError(error_vector, anomaly_loc, anomaly_dur, thresh = 0.99999):
    """
    Using the prediction vector (Input-Predictions) of a predictive model
    we can model this and thus generate a anomaly score from the Gaussian
    fitted to these error values.
    
    """
    data = error_vector

    # Fit a normal distribution to the data:
    mu, std = stats.norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=50, normed=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.arange(xmin, xmax, 1)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()

    anomaly_score = np.zeros_like(data)
    anomaly_score_2 = np.zeros_like(data)
    for i in np.arange(len(data)): #loop through error scores
        for e in np.arange(len(x)): #loop through gaussian indicies
            if np.round(data[i]) == np.round(x[e]): # if error score equals gaussian index
                anomaly_score[i] += p[e] # set anomaly score to gaussian value at this index
                anomaly_score_2[i] += (p[e]-mu).T*(1/std)*(p[e]-mu)

    # Shift to range between 0-1
    anomaly_score = 1-anomaly_score/np.max(anomaly_score)

    # Plot output
    anomaly_score[anomaly_score<thresh]=0
    sim.plot_data((anomaly_score), anomaly_loc, anomaly_dur,title="Extracted anomaly score")
    
    return anomaly_score

