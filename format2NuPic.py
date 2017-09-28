
# coding: utf-8

# In[46]:


# Install Obspy
# https://github.com/obspy/obspy/wiki/Installation-on-Linux-via-Apt-Repository
# with fix https://github.com/obspy/obspy/commit/f4ebf5a3d7c5f809901bb1dd31646b1f76872f47

from pandas import read_csv
from pandas import DataFrame
from pandas import MultiIndex
import wave
import numpy as np
import matplotlib.pylab as plt
from obspy import read

# Earthquake data http://geofon.gfz-potsdam.de/waveform/webservices.php
# http://geofon.gfz-potsdam.de/fdsnws/dataselect/1/query?net=GE&sta=BKB&cha=BHZ&starttime=2011-03-11T06:00:00Z&endtime=2011-03-11T06:05:00Z
# Requests 300 seconds of data from 0600 UTC on 11 March 2011, for the BHZ channel at GEOFON station Balikpapan, Kalimantan

def main(values, typ, FILE_PATH_,FILE_NAME_):
    """
    Import datasets and convert into csv file as requested by Numenta
    
    """
    # file type search
    if typ == "mseed": # earthquake data
        st = read(FILE_PATH_+FILE_NAME_)
        data = st[0].data
    
    elif typ == "wav": # sound wave data
        st = read_csv(FILE_PATH_+"set_b.csv") #list on wave names
        spf = wave.open(FILE_PATH_+"set_b/" +"Bunlabelledtest__101_1305030823364_A.wav")

        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        data = np.fromstring(signal, 'Int16')
        
    else: # simulated data
        data = values

    # Load data into dataframe
    #columns = MultiIndex.from_tuples(list(zip(["timestamp", "value"],["float", "float"],["T", ""])))
    #signal_df = np.array((np.arange(data.size), data)).T
    #signal_df = DataFrame(signal_df, columns=columns)
    
    plt.plot(data)
    plt.title(st[0].stats.starttime)
    plt.show()

    #signal_df.to_csv("/home/codepan1/RestRunnerTestData/earthquake.csv")


values=None
typ="mseed"
FILE_PATH_ = "/home/codepan1/Downloads/"
FILE_NAME_="fdsnws.mseed"
main(values,typ,FILE_PATH_,FILE_NAME_)


# In[52]:


FILE_PATH_ = "/home/codepan1/Downloads/"
FILE_NAME_="fdsnws.mseed"
#st = read(FILE_PATH_+FILE_NAME_)
#data = st[0].data
start=st[0].stats.starttime
end=st[0].stats.endtime
end.datetime
st[0].stats

