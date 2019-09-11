# TODO : Only get data with same SSID as current SSID for fitting

import sys

sys.path.append('/home/pi/Git Repo/data_collector')
sys.path.append('/home/pi/Git Repo/algorithm')

import algorithm.mnist_neural_network as gateway
import data_collector.mnist_data_collector as sensor

import os
import argparse
import logging.config
import yaml
import importlib
import json
import random
import multiprocessing
import time
import sys
import statistics as stats
import csv
import numpy as np
from scipy import stats as spstats
import csv
import inspect
import prediction_utils as predict

from utils.dependency_handler import DependencyHandler
from utils.dynamo_utils import Table
from utils import bluetootch_utils as BT

parser = argparse.ArgumentParser()
parser.add_argument("cfg_file_path", help="The path of configuration file")
parser.add_argument("pi_name", help="The name of this pi")
parser.add_argument("epochs", help="The name of this pi")
parser.add_argument("batch_size", help="The name of this pi")
parser.add_argument("-v", "--verbosity", action="store_true", help="The log level")
parser.add_argument("-l", "--logconfig", default="./configuration/logconfig.ini", help="The log configuration file")
args = parser.parse_args()

global_config = {}
with open(args.cfg_file_path) as f:
	global_config = yaml.load(f)
	
my_config 	= global_config.get(args.pi_name)
role 		= my_config.get("role")
class_path 	= my_config.get("classPath")
class_name 	= my_config.get("className")
down_addr 	= my_config.get("downStream")

epochs = int(args.epochs)
batch_size = int(args.batch_size)

csv_write_file = 'ks_data_wifi_nn_remote.csv'
write_file = 'mnist_experiment_lte_remote.csv'

gateway_node = gateway.MNISTNetwork()
gateway_node.init(args.pi_name, my_config)

sensor_node = sensor.MNISTDataCollector()
sensor_node.init('RPiB', my_config)

table = Table('mnresult-lambda',['environment', 'S'])
# Time records

sensing_times 	= []
algo_times 		= []
get_db_times 	= []
push_db_times 	= []

# Create two processes
# 	1. Gets data from bluetooth and adds it to the stack
# 	2. Uses data from bluetooth to run algorithm and send data to DynamoDB

ctr = 0
res = 1000

windows = []
tx_times = []
min_times = 100000
threshold = 0.3
window_size = 20
linspace_size = 1000
res = 10000

time_threshold = 50
ks_threshold = 0.1

batch_sizes = [100,200,300,400,500,600,700,800,900,1000]
iterations  = [20,30,40,50,60,70,80,90,100]
max_runs 	= 10 
distribs = [spstats.f,spstats.fisk,spstats.burr12,spstats.exponnorm,spstats.johnsonsu,spstats.norminvgauss]
for run in range(max_runs):	
	sensor_node.run()
	rx_data = sensor_node._data #BT.listenOnBluetooth(1)

	print("Begin")

	#ssid = os.popen("iwconfig wlan0 | grep 'ESSID' | awk '{print $4}' | awk -F\\\" '{print $2}'").read()
		
	t_start = time.time()
	current_data 	= rx_data
	#print('Current Data is ',current_data)
	total_bt_time 	= time.time()
#current_data 	= current_data[0]
	try:
		id = current_data.get('from_pi')
	except:
		print('effed up')
	else:
		print("Processing Data from ",id)
		if(current_data):
			print('Run : ',run)
			# Get Updated Weights from DynamoDB
			
			t_get_weights_start = time.time()
			try:
				record = table.getItem({'environment' : 'roomA_mnist_1000'})
				weights = record['weights']
			except:
				weights = '1'
			t_get_weights_end = time.time()
			print('Rx Weight Time : ',(t_get_weights_end-t_get_weights_start))
			
			# Local Computation
			weights = np.array(weights)
			t_algo_start = time.time()
			gateway_node.run(train_data=[current_data],iterations=epochs,batch_size=batch_size,weights=weights)
			t_algo_end = time.time()
			print("Algorithm Time : ",(t_algo_end-t_algo_start))

#------------------------------------------------------------------------------------------------------------------------------------------------
			# Tx Time Prediction
			#
			# Additional variables : Number of required time data, threshold time, window size, change detection threshold
			#
			# Check if total number of stored times is enough
			
			if len(tx_times)>min_times:
			# If enough:
			# Meta algo : Fit all stored times to a distribution
				t_tx = time.time()
				gateway_node.send(down_addr=down_addr, bt_time=total_bt_time)
				t_end = time.time()
				
				tx_times.append(t_end-t_tx)
				
				min_ks = 1
				
				for dist in distribs:
					model_params = dist.fit(tx_times)
					ks_stat,p_val = spstats.kstest(tx_times,dist.name,[*dist.fit(tx_times)])
					if(ks_stat<min_ks):
						min_ks = ks_stat
						optimal_params = model_params
						optimal_dist = dist
						prob_dist = dist.pdf(np.linspace(0, linspace_size, res),*optimal_params)
						print('Minimum KS is ',min_ks,'for distribution ',dist.name)

				tx_probs = []
				for time_val in tx_times:
					tx_probs.append(predict.get_prob(time_val,prob_dist,linspace_size,res))
				tx_probs_norm = (np.array(tx_probs)/max(tx_probs))*100
			
			# Map trend using stored times --> Kalman Filter
				trend = predict.kalman_filter(tx_probs_norm,200)
				
			# Detect change?
				avg_log,change_var,change_detected = predict.detect_change(trend,threshold,window_size)
				avg_log = np.array(avg_log)/max(avg_log)
				change_var = np.array(change_var)/max(change_var)
				wt = 0.5
				combination = wt*avg_log+(1-wt)*change_var
				change_pt = combination[-window_size-1]
				#print('Combination ',combination)
				#print('Avg Log ',avg_log)
				#print('Var Log ',change_var)
				
				line = [run, int(time.time()), optimal_dist.name, min_ks, optimal_params, t_end-t_tx, change_pt]
				with open(csv_write_file, 'a') as writeFile:
					writer = csv.writer(writeFile)
					writer.writerow(line)
				
			# If change detected:
				print(change_pt)
				if(change_pt>0.8 and min_ks<ks_threshold):
					print('Change Detected')
					
				# Select all points before change --> Assign to a distribution, store parameters
					windows.append(tx_times[:-window_size-1])
					
				# Select all points after change --> Assign to a distribution, store parameters, treat as new current distribution
					tx_times = []
					
				# Predict expected value of distribution
					expected = []
					for i in range(0,len(prob_dist),10):
						expected.append(prob_dist[i]*i/10)
					prediction = sum(expected)
					
				# If sample > threshold:
					if(prediction>time_threshold):
					
				# Local computation with 50% probability
						if(np.random.rand()<0.5):
							print('Local Computation')
							
				# Remote computation
						else:
							print('Remote Computation')
							
				time.sleep(90)
			# If not enough:
			else:
			# 			Append to list of stored weights
				t_tx = time.time()
				gateway_node.send(down_addr=down_addr, bt_time=total_bt_time)
				t_end = time.time()
				print("Tx time : ",t_end-t_tx)
				
				# Write to CSV
			
				get_weights_time 	= (t_get_weights_end-t_get_weights_start)
				algo_time 			= (t_algo_end-t_algo_start)
				tx_time 			= (t_end-t_tx)
				line = [run,int(time.time()),get_weights_time,algo_time,tx_time]
				with open(write_file, 'a') as writeFile:
						writer = csv.writer(writeFile)
						writer.writerow(line)
						writeFile.close()
			
				tx_times.append(t_end-t_tx)
			ctr+=1
			run += 1
			
#--------------------------------------------------------------------------------------------------------------------------------------------------
			gateway_node.cleanup()

			print('Waiting...')
			print( )
			time.sleep(75)
		else:
			print('No data received')
			time.sleep(5)



