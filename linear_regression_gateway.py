import sys

sys.path.append('/home/pi/Git Repo/data_collector')
sys.path.append('/home/pi/Git Repo/algorithm')

import algorithm.linear_regression_algorithm as gateway

import argparse
import logging.config
import time
import yaml
import importlib
import json
import random
import multiprocessing
import time

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
# Create two processes
# 	1. Gets data from bluetooth and adds it to the stack
# 	2. Uses data from bluetooth to run algorithm and send data to DynamoDB

def put_data(q):
	while(1):
		bt_time, recv_data = BT.listenOnBluetooth(1)
		q.put([recv_data,bt_time])
	return recv_data
	
def print_stack(q):
	while(1):
		current_data 	= q.get()
		total_bt_time 	= current_data[1]
		current_data 	= current_data[0]
		if(current_data):
			print("Rx Data!")
			w_1 = 0
			w_2 = 0
			gateway_node.run(train_data=current_data,w_1=w_1,w_2=w_2)
			gateway_node.send(down_addr=down_addr, bt_time=total_bt_time)
			gateway_node.cleanup()
			time.sleep(5)
		else:
			print('No data received')
			time.sleep(5)
#gateway_node.run(train_data=train_data,w_1=w_1,w_2=w_2)
#gateway_node.send(down_addr=down_addr, bt_time=total_bt_time)
#gateway_node.cleanup()

print("Creating processes")

q = multiprocessing.Queue()

proc_1 = multiprocessing.Process(target=put_data,args=(q,))
proc_2 = multiprocessing.Process(target=print_stack,args=(q,))

proc_1.start()
proc_2.start()

proc_1.join()
proc_2.join()

print("Done!")