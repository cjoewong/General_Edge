#!/usr/bin/env python3

import argparse
import logging.config
import time
import yaml
import importlib
import sys
import json

from utils.dependency_handler import DependencyHandler
from utils.dynamo_utils import Table

sys.path.append('/home/pi/Git Repo/data_collector')
sys.path.append('/home/pi/Git Repo/algorithm')

import linear_regression_data_collector as sensor
import linear_regression_algorithm as gateway

def init_logger(config_path, verbosity):
    """
    Init the global logger

    Param(s):
        config_path  The logger configuration file path
        verbosity    The logger level
    """
    logging.config.fileConfig(config_path)
    logger = logging.getLogger()
    if verbosity:
        logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    print('Start...')

    # Setup the command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file_path", help="The path of configuration file")
    parser.add_argument("pi_name", help="The name of this pi")
    parser.add_argument("-v", "--verbosity", action="store_true", help="The log level")
    parser.add_argument("-l", "--logconfig", default="./configuration/logconfig.ini", help="The log configuration file")
    args = parser.parse_args()

    init_logger(args.logconfig, args.verbosity)

    logger = logging.getLogger()
    logger.info("Start parsing the configuration_file....")

    # Read the configuration file
    logger.info("Parsing configuration file...")
    global_config = {}
    with open(args.cfg_file_path) as f:
        global_config = yaml.load(f)

    # Resolve Pi's dependencies - Not needed
    dependency_handler = DependencyHandler(global_config)

    # Schedule Pi's and the Pi will only be scheduled when all its dependencies
    # are resolved
    train_data = []
    total_bt_time = 0

    # Run it's main function
    my_config = global_config.get(args.pi_name)
    role = my_config.get("role")

    # Use reflection to dynamically new instance
    class_path = my_config.get("classPath")
    class_name = my_config.get("className")
    #m = importlib.import_module(class_path)
    #clz = getattr(m, class_name)

	# TO BE CHANGED WITHIN THE CONFIG FILE
    down_addr = 'sensingdata_A'

    logger.info("Run worker...")

    print('Running...')
	
    sensor_node = sensor.LinearRegressionDataCollector(args.cfg_file_path)
    gateway_node = gateway.LinearRegression()
    
    gateway_node.init(args.pi_name, my_config)
    sensor_node.init(args.pi_name, my_config)

    # This should loop
    loop_count = 0
    max_loop_count = 5
    while(loop_count<max_loop_count):	
        t1 = time.time()
        table = Table('testresult')
        record = table.getItem({'environment' : 'roomA', 'sensor' : 'sensorA&B&C'})
        w_1 = record['w_1']
        w_2 = record['w_2']
        print('weights')
        print(w_1,w_2)
        
        sensor_node.run()
        train_data = sensor_node.send(down_addr=down_addr, bt_time=total_bt_time)
        sensor_node.cleanup()

        print('Train Data Start')
        #print( )
        #print(train_data.get("data"))
        #print( )
        print('Train Data End')
	
	    # Simulate Transmission delay
	    # Store training data in a variable
	
        gateway_node.run(train_data=train_data,w_1=w_1,w_2=w_2)
        gateway_node.send(down_addr=down_addr, bt_time=total_bt_time)
        gateway_node.cleanup()
    
        print('Check DynamoDB')
	
        t2 = time.time()

    logger.info("End....")
    print("End...")
