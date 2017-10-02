
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from datetime import timedelta 
import pandas as pd

def get_data(n=0,datalabels=["timestamp","consumption"]):
    
    """
    Generate simulated data set
    
    Anomaly locations should be at multiples of 24 to keep phase constant
    
    """
    N=10000
    Start = 8000
    A_base = 20
    dur_base = 24
    f_week = (2*np.pi)/ 168  # weekly period in hours
    f_day = (2*np.pi) / 24   # daily period in hours
    time_step = 1
    
    data = np.zeros(N)
    
    # Base signal
    t_xrange = np.arange(0,N,time_step)
    data = (A_base/2)*np.sin(f_day*t_xrange) +(A_base/2)*np.sin(f_week*t_xrange)     

    # Sudden amplitude shift *for daily sine
    A = 5                             # amplitude shift
    A_loc = 8208                      # location
    A_dur = dur_base                  # anomaly duration
    t_Arange = np.arange(A_loc,A_loc+A_dur) 
    data[t_Arange] = (A/2)*np.sin(f_day*t_Arange) +(A_base/2)*np.sin(f_week*t_Arange) 

    # Gradual amplitude shift *for both daily sine
    As = 2                            # amplitude shift
    As_loc = 8472                     # location
    As_dur = dur_base*3               # anomaly duration
    t_Asrange = np.arange(As_loc,As_loc+As_dur,1)
    As_shift = np.linspace(A_base,A_base*As,As_dur)
    data[t_Asrange] = (As_shift/2)*np.sin(f_day*t_Asrange) +(A_base/2)*np.sin(f_week*t_Asrange) 

    # Sudden Frequency shift *for daily sine
    f = (2*np.pi) / 5                             # frequency shift
    f_loc = 8760                      # location
    f_dur = dur_base                  # anomaly duration
    t_frange = np.arange(f_loc,f_loc+f_dur)
    data[t_frange] = (A_base/2)*np.sin(f*t_frange) +(A_base/2)*np.sin(f_week*t_frange) 

    # Gradual frequency shift
    fs = 1 / 12                           # frequency shift
    fs_loc = 9000                     # location
    fs_dur = dur_base*3               # anomaly duration
    t_fsrange = np.arange(fs_loc,fs_loc+fs_dur,1)
    fs_shift = np.linspace(f_day/(2*np.pi),fs,fs_dur)
    data[t_fsrange] =(A_base/2)*np.sin(2*np.pi*(t_fsrange + f_day)*t_fsrange) + (A_base/2)*np.sin(f_week*t_fsrange)

    # Sudden Phase Shift
    p = 0.5*np.pi
    p_loc = 9240
    p_dur = dur_base
    t_prange = np.arange(p_loc,p_loc+p_dur)
    data[t_prange] = (A_base/2)*np.sin(f_day*t_prange + p) +(A_base/2)*np.sin(f_week*t_prange) 

    # Gradual Phase Shift
    ps = 0.5*np.pi
    ps_loc = 9480
    ps_dur = dur_base
    t_psrange = np.arange(ps_loc,ps_loc+ps_dur)
    ps_shift =  np.linspace(1,ps,ps_dur)
    data[t_psrange] = (A_base/2)*np.sin(f_day*t_psrange + ps_shift) +(A_base/2)*np.sin(f_week*t_psrange) 

    # Add noise
    N_s = n                   # noise magnitude percentage
    data = data + np.random.normal(0,N_s*A_base,N)

    # Missing data
    M_loc = 9800                        # location
    M_dur = dur_base                    # anomaly duration
    data[M_loc:M_loc+M_dur] = 0

    # Convert data indicies into hours '%m/%d/%y %H:%M'
    data_start = "01/03/2017 00:00"
    dates = []
    for hour in range(N):
        dates.append(datetime.strptime(data_start, '%m/%d/%Y %H:%M') + timedelta(hours=hour)) 
    
    # Output the anomaly locations and durations (for plotting later)
    anomaly_loc = [A_loc, As_loc, f_loc, fs_loc, p_loc, ps_loc, M_loc]
    anomaly_dur = [A_dur, As_dur, f_dur, fs_dur, p_dur, ps_dur, M_dur]
    
    # Limit data to 3 decimal places
    data = np.round(data,3)
    
    return data, anomaly_loc, anomaly_dur, dates


def plot_data(data, anomaly_loc, anomaly_dur,title, Start=8000):
    
    # Arrays for plotting
    anomaly_range = []
    for i in range(np.size(anomaly_loc)):
        anomaly_range.append(np.arange(anomaly_loc[i],anomaly_loc[i]+anomaly_dur[i]))

    plt.figure(1)
    plt.plot(data, label = 'Data',c='k', zorder=0)
    plt.scatter(range(100),np.zeros(100) ,s=5, label = 'Normal' ,zorder=10)

    plt.scatter(anomaly_range[0] -Start, np.zeros_like(anomaly_range[0]),s=5, label = 'Sudden Amplitude shift' ,zorder=10)
    plt.scatter(anomaly_range[1] -Start, np.zeros_like(anomaly_range[1]),s=5, label = 'Gradual Amplitude shift', zorder=10)
    plt.scatter(anomaly_range[2] -Start, np.zeros_like(anomaly_range[2]),s=5, label = 'Sudden Frequency shift', zorder=10)
    plt.scatter(anomaly_range[3] -Start, np.zeros_like(anomaly_range[3]),s=5, label = 'Gradual Frequency shift', zorder=10)
    plt.scatter(anomaly_range[4] -Start, np.zeros_like(anomaly_range[4]),s=5, label = 'Sudden Phase shift', zorder=10)
    plt.scatter(anomaly_range[5] -Start, np.zeros_like(anomaly_range[5]),s=5, label = 'Gradual Phase Shift', zorder=10)
    plt.scatter(anomaly_range[6] -Start, np.zeros_like(anomaly_range[6]),s=5, label = 'Missing data', zorder=10)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Time axis")
    plt.ylabel("Amplitude axis")
    plt.title(title)
    plt.show()
    
def plot_anomalies(data, anomaly_loc, anomaly_dur, Start=8000, buffer = 100):
    
    
    names = ['Sudden Amplitude','Gradual Amplitude','Sudden Frequency','Gradual Frequency',
            'Sudden Phase', 'Gradual Phase']
    plt.figure(2)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.plot(data[anomaly_loc[i]-buffer:anomaly_loc[i]+anomaly_dur[i]+buffer])
        plt.title(names[i])
        plt.ylim([-50,50])
        plt.xlim([0,300])
        plt.axis('off')
        
    plt.show()
    
#data, anomaly_loc, anomaly_dur, dates = get_data(n=0,datalabels=["dttm","value"])
#plot_data(data[8000:], anomaly_loc, anomaly_dur,title = "simulated data")
#plot_anomalies(data, anomaly_loc, anomaly_dur)



# In[ ]:


# plt.cla
# f_week = (2*np.pi)/ 168  # weekly period in hours
# f_day = (2*np.pi) / 24 
# A_base = 20
# dur_base = 24
# # Gradual frequency shift
# fs = 1 / 12                           # frequency shift
# fs_loc = 9000                     # location
# fs_dur = dur_base*3               # anomaly duration
# t_fsrange = np.linspace(fs_loc,fs_loc+fs_dur,fs_dur)
# fs_shift = np.linspace(f_day/(2*np.pi),fs,fs_dur)
# t = np.linspace(0,1,fs_dur)

# plt.plot((A_base/2)*np.sin(2*np.pi*(t_fsrange + fs_shift)*t_fsrange))# + (A_base/2)*np.sin(f_week*t_fsrange))
# plt.show()


# In[ ]:


# A_base = 20
# T_week = (2*np.pi)/ 168  # weekly period in hours
# T_day = (2*np.pi) / 24   # daily period in hours
# dur_base = 24
# # Gradual frequency shift
# fs = 1.5                      # frequency shift
# fs_loc = 9000                     # location
# fs_dur = dur_base             # anomaly duration
# fs_range =  np.arange(fs_loc,fs_loc+fs_dur)
# fs_shift =  np.arange(1,fs,((fs-1)/fs_dur))
# plt.plot((A_base/2)*np.sin((T_day*(fs_range))) +(A_base/2)*np.sin((T_week*(fs_range/fs_shift))))
# plt.show()

# fs_shift.shape

