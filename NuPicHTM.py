
# coding: utf-8

# In[1]:


# OPF Swarming to Optimize Algorithm for Predictions (and the simulated data)

"""  Command line password edit for MySQL """
# export NTA_CONF_PROP_nupic_cluster_database_passwd=MySQL123

""" Run swarm """
# ~/nupic/scripts/run_swarm.py --overwrite ~/myswarm/data/search_def.json 


# In[2]:


# Set up opf model and extract anomalies from data

import matplotlib.pylab as plt
import yaml
import csv
import datetime
from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.model_factory import ModelFactory
import importlib
import SimAnomalyDataset as sim

modelParams = importlib.import_module("model_params").MODEL_PARAMS # best_model_params / model_params

# Create dataset
datalabels=["dttm","value"]
data, anomaly_loc, anomaly_dur, dates = sim.get_data(n=0,datalabels=datalabels)
      


# In[3]:



# Create OPF Model & Load parameters into model space
model = ModelFactory.create(modelParams)

# What to predict?
model.enableInference({'predictedField': datalabels[1]})

# Open the file to loop over each row to feed model
output = []
anomaly_score = []
prediction = []
confidence = []
anomaly_Likelihood = []
anomaly_logLikelihood = []
anomaly_likelihood_helper = anomaly_likelihood.AnomalyLikelihood()
input_data = []
with open ("sim_data.csv") as fileIn:
    reader = csv.reader(fileIn)
    # The first two rows are not data, but we'll need the field names when passing data into the model.
    headers = reader.next()
    reader.next()
    reader.next()

    # loop through data rows
    for record in reader:
        # Save input data for plotting
        input_data.append(record)
        
        # Create a dictionary with field names as keys, row values as values.
        modelInput = dict(zip(headers, record))
        
        # Convert string consumption to float value.
        modelInput[datalabels[1]] = float(modelInput[datalabels[1]])
        
        # Convert timestamp string to Python datetime.
        modelInput[datalabels[0]] = datetime.datetime.strptime(
          modelInput[datalabels[0]], "%Y-%m-%d %H:%M:%S")
        
        # Push the data into the model and get back results.
        result = model.run(modelInput)
        
        # Save predicition history in new file
        output.append(result.inferences['multiStepBestPredictions'][1])
        
        anomaly_score_r = result.inferences["anomalyScore"]
        
        prediction_r = result.inferences["multiStepBestPredictions"][1]
        
        confidence.append(result.inferences["multiStepPredictions"][1][prediction_r])

        anomaly_Likelihood_r = anomaly_likelihood_helper.anomalyProbability(
            modelInput[datalabels[1]], anomaly_score_r, modelInput[datalabels[0]]
        )
        
        anomaly_logLikelihood_r = anomaly_likelihood_helper.computeLogLikelihood(anomaly_Likelihood_r)
        
        #store for later
        anomaly_score.append(anomaly_score_r)
        prediction.append(prediction_r)
        anomaly_Likelihood.append(anomaly_Likelihood_r)
        anomaly_logLikelihood.append(anomaly_logLikelihood_r)


# In[ ]:


import numpy as np
import evaluatePredictions as evalPred

# slight data transformation
a = np.asarray(input_data)
a = a[:,2].astype(np.float)

metric = "RMSE"
OUT = evalPred.main(test=a[8000:], pred=prediction[8000:], metric = "MSE")

#RMSE plot
thresh_min = 200
thresh_max = 500
to_plot=np.asarray(OUT)
to_plot[to_plot<thresh_min]=0
to_plot[to_plot>thresh_max]=thresh_max
to_plot = (to_plot/np.max(to_plot)) #remove normalizer whilst thresholding

sim.plot_data((to_plot), anomaly_loc, anomaly_dur,title=metric)

#anomaly likelihoods
sim.plot_data((anomaly_score), anomaly_loc, anomaly_dur,title="anomaly_score")


# In[ ]:


pred=a[8000:]-prediction[8000:]
evalPred.GaussianPredError(pred,anomaly_loc, anomaly_dur,thresh=0.97)

