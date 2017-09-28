
# coding: utf-8

# In[1]:


# Install Obspy
# https://github.com/obspy/obspy/wiki/Installation-on-Linux-via-Apt-Repository

from obspy import read
FILE_PATH_ = "/home/codepan1/Downloads/Package_1506336094435.mseed"
st = read(FILE_PATH_)
st 

