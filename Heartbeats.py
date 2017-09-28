
# coding: utf-8

# In[16]:


from pandas import read_csv
from pandas import DataFrame
from pandas import MultiIndex
import wave
import numpy as np
import matplotlib.pylab as plt

# Import Heartbeats example and convert into csv 
FILE_PATH_ = "/home/codepan1/Downloads/"
st = read_csv(FILE_PATH_+"set_b.csv")
st
spf = wave.open(FILE_PATH_+"set_b/" +"Bunlabelledtest__101_1305030823364_A.wav")

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

# Load data into dataframe
columns = MultiIndex.from_tuples(list(zip(["timestamp", "value"],["float", "float"],["T", ""])))
signal_df = np.array((np.arange(signal.size), signal)).T
signal_df = DataFrame(signal_df, columns=columns)

signal_df

signal_df.to_csv("/home/codepan1/RestRunnerTestData/Heartbeats/heartbeats_singlefile.csv")


# In[23]:


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.show()

