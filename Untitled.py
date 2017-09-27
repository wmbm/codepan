
# coding: utf-8

# In[7]:


from pandas import read_csv
import matplotlib.pylab as plt
PATH  = "/home/codepan1/RestRunnerTestData/NAB/results/alpha/realTweets/"
name = read_csv(PATH + "alpha_Twitter_volume_AAPL.csv" )


plt.plot(name["value"])
plt.show()

