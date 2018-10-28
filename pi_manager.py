#!/usr/bin/env python3

import argparse
import logging.config
import time
import yaml
import importlib

import utils.bluetootch_utils as BT
from utils.dependency_handler import DependencyHandler


def init_logger(config_path, verbosity):
    """
    Init the global logger

    @param: verbosity The logger level
    """
    logging.config.fileConfig(config_path)
    logger = logging.getLogger()
    if verbosity:
        logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    print('Start...')
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file_path",
                        help="The path of configuration file")
    parser.add_argument("pi_name",
                        help="The name of this pi")
    parser.add_argument("-v", "--verbosity",
                        action="store_true",
                        help="The log level")
    parser.add_argument("-l", "--logconfig",
                        default="./configuration/logconfig.ini",
                        help="The log configuration file")
    args = parser.parse_args()

    init_logger(args.logconfig, args.verbosity)

    logger = logging.getLogger()
    logger.info("Start....")
    # Read the configuration file
    logger.info("Parsing configuration file...")
    global_config = {}
    with open(args.cfg_file_path) as f:
        global_config = yaml.load(f)

    # Resolve pis' dependencies
    dependency_handler = DependencyHandler(global_config)

    # schedule
    train_data = []
    total_bt_time = 0
    total_data_size = 0
    while not dependency_handler.dependency_resolved(args.pi_name):
        logger.info("Waiting for Pi-{0}'s dependencies...".format(args.pi_name))
        bt_time, recv_data = BT.listenOnBluetooth(1)
        total_bt_time += bt_time
        total_data_size += recv_data.nbytes
        train_data.append(recv_data)
        from_pi = recv_data.get('from_pi')
        dependency_handler.add_resolved_dependency(from_pi, args.pi_name)

    # run it's main function
    my_config = global_config.get(args.pi_name)
    role = my_config.get("role")
    if role == "DataCollector":
        pass
    elif role == "Algorithm":
        pass
    else:
        raise RuntimeError("Error role of pi-{0}".format(args.pi_name))
    # Use reflection to dynamically new instance
    class_path = my_config.get("classPath")
    class_name = my_config.get("className")

    m = importlib.import_module(class_path)

    clz = getattr(m, class_name)

    try:
        down_stream = my_config.get("downStream")
        print("down_stream:" + down_stream)
        if down_stream is None:
            raise RuntimeError()
        down_addr = global_config.get(down_stream).get("btAddress")
    except Exception:
        logger.warn("Pi-{0} has no down stream, right?".format(args.pi_name))
        down_addr = ""

    logger.info("Run worker...")

    t1 = time.time()
    worker = clz()
    worker.init(args.pi_name, my_config)
    worker.run(train_data=train_data)
    worker.send(down_addr=down_addr, bt_time=total_bt_time, origin_data_size=total_data_size)
    worker.cleanup()
    t2 = time.time()

    logger.info("End....")
    print("End...")
