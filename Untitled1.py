
# coding: utf-8

# In[22]:


from pandas import read_csv
import wave
import numpy as np
import matplotlib.pylab as plt
FILE_PATH_ = "/home/codepan1/Downloads/"
st = read_csv(FILE_PATH_+"set_b.csv")
st
spf = wave.open(FILE_PATH_+"set_b/" +"Bunlabelledtest__101_1305030823364_A.wav")

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

plt.plot(signal)
plt.show()

