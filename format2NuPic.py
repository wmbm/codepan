
# coding: utf-8

# In[ ]:


# Install Obspy
# https://github.com/obspy/obspy/wiki/Installation-on-Linux-via-Apt-Repository
# with fix https://github.com/obspy/obspy/commit/f4ebf5a3d7c5f809901bb1dd31646b1f76872f47

import pandas as pd
import wave
import numpy as np
import matplotlib.pylab as plt
from obspy import read

# Earthquake data http://geofon.gfz-potsdam.de/waveform/webservices.php
# http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query?net=GE&sta=BKB&cha=BHZ&starttime=2011-03-11T06:00:00Z&endtime=2011-03-11T06:05:00Z
# Requests 300 seconds of data from 0600 UTC on 11 March 2011, for the BHZ channel at GEOFON station Balikpapan, Kalimantan

def main( FILE_PATH_,FILE_NAME_,values=None, typ="mseed", output_type="HTM",datalabels=["timestamp","consumption"]):
    """
    Import datasets and convert into csv file as requested by Numenta (NUPIC)
    
    limit : downloaded earthquake file is very large - limit data points but include a earthquake
    
    Parameters
    ----------
    values : optional value imports simulated data
    output_type : specify the format of the output file for NAB or HTM or other
    
    """
    
    OUTPUT_PATH_ = "/home/codepan1/RestRunnerTestData/earthquake.csv"
    
    
    # File type search and onfigure
    if typ == "mseed": # earthquake data
        st = read(FILE_PATH_+FILE_NAME_)
        data = st[0].data
        info = st[0].stats
    
    elif typ == "wav": # sound wave data (Heart Beats)
        st = pd.read_csv(FILE_PATH_+"set_b.csv") #list on wave names
        spf = wave.open(FILE_PATH_+"set_b/" +"Bunlabelledtest__101_1305030823364_A.wav")

        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        data = np.fromstring(signal, 'Int16')
        
    else: # simulated data
        data = values

    # Load data into dataframe
    if output_type == "NAB":
        columns = pd.MultiIndex.from_tuples(list(zip(["timestamp", "value"],["float", "float"],["T", ""])))
        signal_df = np.array((np.arange(data.size), data)).T
        signal_df = pd.DataFrame(signal_df, columns=columns)
        signal_df.to_csv(OUTPUT_PATH_)
    if output_type == "HTM":
        dates = np.arange(len(data))
        data_pd = pd.DataFrame(data=np.array((dates, list(data))).T, index=range(len(dates)),
                       columns=[datalabels[0],datalabels[1]])
        data_pd.to_csv(OUTPUT_PATH_ ,columns=[datalabels[0],datalabels[1]])
    
    return info
#     plt.plot(data)
#     plt.title(st[0].stats.starttime)
#     plt.show()

    


# In[ ]:


# FILE_PATH_ = "/home/codepan1/Downloads/"
# FILE_NAME_ = "fdsnws.mseed"
# limit = 20000000
# st = read(FILE_PATH_+FILE_NAME_)
# data = st[0].data[limit:]
# tr = st[0] 
# st.plot()


# In[ ]:


# # extract datetime to time
# dates = []
# for t in tr.times():
#     dates.append(tr.stats.starttime + t)


# In[ ]:


# """
# Write to NUMENTA style csv file (Also for Numenta NAB)
# """

# # Convert anomaly locations to binary array
# labels = np.zeros_like(dates)
# for i in anomaly_loc:
#     labels[i] = 1
    
# # Pad anomaly scores to zero during training
# full_anomaly_scores = np.zeros_like(dates)
# full_anomaly_scores[8000:] = anomaly_score

# # Write data to CSV file
# datacsv = DataFrame()
# datacsv["timestamp"] = dates
# datacsv["value"] = np.round(data,3)
# datacsv["anomaly_score"] = full_anomaly_scores
# datacsv["label"] = labels

# ## Write CSV to folder
# #datacsv.to_csv(path_or_buf="/home/codepan1/RestRunnerCode/alpha_Twitter_volume_AAPL.csv")
# #series = read_csv("/home/codepan1/RestRunnerCode/alpha_Twitter_volume_AAPL.csv")


