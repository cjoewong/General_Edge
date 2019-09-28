# TODO : Only get data with same SSID as current SSID for fitting

import sys

sys.path.append('/home/pi/Git Repo/data_collector')
sys.path.append('/home/pi/Git Repo/algorithm')

import algorithm.linear_regression_algorithm as gateway
import data_collector.linear_regression_data_collector as sensor

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

gateway_node = gateway.LinearRegression()
gateway_node.init(args.pi_name, my_config)

sensor_node = sensor.LinearRegressionDataCollector(my_config)
sensor_node.init('RPiB', my_config)

table = Table('testresult')
# Time records

sensing_times 	= []
algo_times 		= []
get_db_times 	= []
push_db_times 	= []

# Create two processes
# 	1. Gets data from bluetooth and adds it to the stack
# 	2. Uses data from bluetooth to run algorithm and send data to DynamoDB

def put_data(q,flag):
	while(1):
		sensor_node.run()
		rx_data = sensor_node._data #BT.listenOnBluetooth(1)
		q.put(rx_data)
		if(flag.value==1):
			print("I'm done too.")
			break
		time.sleep(15)
	
def print_stack(q,flag):
	ctr = 0
	res = 1000
	
	windows = []
	tx_times = []
	min_times = 50
	threshold = 0.3
	window_size = 20
	linspace_size = 1000
	res = 10000
	
	time_threshold = 50
	ks_threshold = 0.1
	
	distribs = [spstats.f,spstats.fisk,spstats.burr12,spstats.exponnorm,spstats.johnsonsu,spstats.norminvgauss]
	
	run = 0

	print("Begin")
	
	while(1):
	
		table = Table('testresult')
		ssid = os.popen("iwconfig wlan0 | grep 'ESSID' | awk '{print $4}' | awk -F\\\" '{print $2}'").read()
		
		t_start = time.time()
		current_data 	= q.get()
		total_bt_time 	= time.time()
		
		try:
			id = current_data.get('from_pi')
		except:
			print('effed up')
		else:
			print("Processing Data from ",id)
			if(current_data):
				t_get_weights = time.time()
				try:
					w_1 = record['w_1']
					w_2 = record['w_2']
				except:
					w_1 = 1
					w_2 = 1
					
				gateway_node.run(train_data=current_data,w_1=w_1,w_2=w_2)
				gateway_node.send(down_addr=down_addr, bt_time=total_bt_time)
				gateway_node.cleanup()
							
				print('Waiting...')
				print( )
				time.sleep(25)
				ctr+=1
			else:
				print('No data received')
				time.sleep(5)

print("Creating processes")

q = multiprocessing.Queue()
flag = multiprocessing.Value('i',0)

proc_1 = multiprocessing.Process(target=put_data,args=(q,flag))
proc_2 = multiprocessing.Process(target=print_stack,args=(q,flag))

print("Processes started!")
print()
print(time.time())
proc_1.start()
proc_2.start()

if(flag.value==1):
	proc_1.terminate()
	proc_2.terminate()

proc_1.join()
proc_2.join()

print("Done!")