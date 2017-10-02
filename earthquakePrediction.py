
# coding: utf-8

# In[1]:


import NABimplementation as NAB
from pandas import read_csv
import format2NuPic

# Import mseed and convert into csv file
states = format2NuPic.main(FILE_PATH_ = "/home/codepan1/Downloads/", FILE_NAME_ = "Package_1506949297549.mseed")
# Import that csv file
series = read_csv("/home/codepan1/RestRunnerTestData/earthquake.csv", header=0, index_col=0, squeeze=True)

#values = series['value'][2:]
#timestamp = series['timestamp'][2:]
states


# In[2]:


# values = series['consumption'].tolist()
# values = values[2:]
# values = [float(i) for i in values]
values = series["consumption"]
#series


# In[3]:


import numpy as np
import matplotlib.pylab as plt
#to_plot = (values[17000000:20000000])
#to_plot[to_plot<0]=0
#time = np.arange(0,0.05*to_plot.size,0.05)
to_plot = values
to_plot[to_plot<0]=0
plt.plot((to_plot))
plt.xlabel("Time (hours)")
plt.ylabel("Magnitude")
plt.show()

values = to_plot
to_plot.size


# In[20]:


from kerasLSTM import main as LSTM
from NuPicHTM import HTM
# Run detector
detector = "HTM"
if detector == "LSTM":
    out, forecasts2 = LSTM(values,n_test = 500000)
elif detector == "HTM":
    anomaly_score = HTM(values)




# In[18]:



plt.subplot(1,2,1)
plt.plot((values))
plt.xlabel("Seismograph")
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.plot((out))
plt.xlabel("Prediction error")
plt.xticks([])
plt.yticks([])
plt.show()

