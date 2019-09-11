import csv
import numpy as np
from scipy import stats
import inspect

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def get_prob(time,p,linspace_size,res):
    if(time<linspace_size):
        idx = int(time*res/linspace_size)
        to_return = p[idx]
    else:
        to_return = 0
    return to_return

def plot_scatter(series):
    x_ax = np.linspace(0, len(series),len(series))
    plt.scatter(x_ax,series)
    plt.show()
    return 1

def detect_change(input_series,threshold,window_size):
    flag = 2
    avg_log = [0]*len(input_series)
    change_detected = []
    change_var = [0]*len(input_series)
    for idx in range(window_size,len(input_series)-window_size):
        curr_difference = 0
        past_difference = 0
        past_window = input_series[idx-window_size:idx]
        curr_window = input_series[idx:idx+window_size]
        
        curr_var = np.var(curr_window)
        past_var = np.var(past_window)
        
        change_var[idx] = (abs(curr_var-past_var))
        
        for win_idx in range(len(curr_window)-1):
            curr_difference+=curr_window[win_idx]
        avg_diff_curr = curr_difference/window_size
        
        for win_idx in range(len(past_window)-1):
            past_difference+=past_window[win_idx]
        avg_diff_past = past_difference/window_size
        
        avg_diff = abs(avg_diff_curr-avg_diff_past)
        avg_log[idx]=(avg_diff)
        if(avg_diff_curr<avg_diff_past):
            flag == 1
        if(avg_diff_curr>avg_diff_past):
            flag == 2
        if(avg_diff>threshold):
            if(avg_diff_curr<avg_diff_past and flag!=0):
                flag = 0
                change_detected.append(idx)
            if(avg_diff_curr>avg_diff_past and flag!=1):
                flag =1
                change_detected.append(idx)
        
    return avg_log,change_var,change_detected

def plot_slices(change_ssid_known):
    print(change_ssid_known)
    global max_len
    plt.axvspan(0, 0, facecolor='r', alpha=0.2,label='3G')
    plt.axvspan(0, 0, facecolor='g', alpha=0.2,label='WiFi')
    for idx in range(0,len(change_ssid_known)-1):
        curr_id = change_ssid_known[idx][0]
        curr_time = change_ssid_known[idx][1]
        next_time = change_ssid_known[idx+1][1]
        if(curr_id=='S'):
            plt.axvspan(curr_time, next_time, facecolor='g', alpha=0.2)
        if(curr_id=='R'):
            plt.axvspan(curr_time, next_time, facecolor='r', alpha=0.2)
    plt.axvspan(next_time, max_len, facecolor='r', alpha=0.2)
    
def load_data(file_name,ssid_list,lim):
    tx_data = []
    with open(file_name+'.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for row in readCSV:
            if(str(row[0])[0] in ssid_list and float(row[2])<lim):
                tx_data.append(float(row[2]))
    return tx_data

def kalman_filter(mean_diff_log,tuner):
    std_dev = np.std(mean_diff_log)
    data_mean = mean_diff_log[0]
    
    # intial parameters
    n_iter = len(mean_diff_log)
    sz = (n_iter,) # size of array
    z = mean_diff_log # observations (normal about x, sigma=0.1)
    
    Q = np.var(mean_diff_log)//tuner # process variance
    
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    R = std_dev**2 # estimate of measurement variance, change to see effect
    
    # intial guesses
    
    xhat[0] = data_mean
    P[0] = 1.0
    
    for k in range(1,n_iter):
        
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
        
    return xhat