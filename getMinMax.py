import pandas as pd
import numpy as np
import sys


def calcMinMax (filename, ix=None):

    print ix
    ix=int(ix)

    df = pd.read_csv(filename,skiprows=[1,2])

    if "timestamp" in df:
		df = df.set_index(['timestamp']) 

    if ix is not None: 
        df = df.ix[:,ix]

	#print df

        dfstd = df.values.std() # gives overall std  
        #print dfstd
        valmin = df.min(axis=0) - dfstd
        valmax = df.max(axis=0) + dfstd

    else: 
        dfstd = df.std(numeric_only=True) # gives std for each column
        valmin = min(df.min(axis=0,numeric_only=True)) - dfstd
        valmax = max(df.max(axis=0,numeric_only=True)) + dfstd


    #resolution = (valmax-valmin)/1000 #00
    resolution = (valmax-valmin)/100

    #print valmin
    #print valmax
    #print resolution

    return valmin, valmax, resolution 


if __name__ == '__main__':

    filename = "/home/pan/Work/restrunner/playground/data/" + sys.argv[1]
    ix = sys.argv[2]
    valmin, valmax, resolution = calcMinMax (filename,ix)
