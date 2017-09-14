
# coding: utf-8

# In[23]:


from pandas import read_csv
import matplotlib.pylab as plt

BASE_LOC = "/home/codepan1/RestRunnerTestData/NAB/data/artificialWithAnomaly/"
NAME_LIST = ["art_daily_flatmiddle.csv","art_daily_jumpsdown.csv",
             "art_daily_jumpsup.csv","art_daily_nojump.csv",
             "art_increase_spike_density.csv","art_load_balancer_spikes.csv"]

series = read_csv(BASE_LOC + NAME_LIST[5], header=0, index_col=0, squeeze=True)
    
plt.plot(series.values)
plt.show()

