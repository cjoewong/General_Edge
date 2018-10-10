#!/usr/bin/env python3

import argparse
import logging
import time
import yaml

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
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file_path",
                        help="The path of configuration file")
    parser.add_argument("pi_name",
                        help="The name of this pi")
    parser.add_argument("-v", "--verbosity",
                        action="store_true",
                        help="The log level")
    parser.add_arguemnt("-l", "--logconfig",
                        default="./configuration/logconfig.ini",
                        help="The log configuration file")
    args = parser.parse_args()

    init_logger(args.logconfig, args.verbosity)

    logger = logging.getLogger()
    logger.info("Start....")
    # Read the configuration file
    logger.info("Parsing configuration file...")
    global_config = yaml.load(args.cfg_file_path)

    # Resolve pis' dependencies
    dependency_handler = DependencyHandler(global_config)

    # schedule
    while not dependency_handler.dependency_resolved(args.pi_name):
        logger.info("Waiting for Pi-{0}'s dependencies...".format(args.pi_name))
        bt_time, recv_data = BT.listenOnBluetooth(1)
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

    # Use reflection to dynamically new isntance
    class_path = my_config.get("class_path")
    class_name = my_config.get("class_name")

    m = __import__(class_path)
    clz = m.getattr(m, class_name)

    logger.info("Run worker...")
    t1 = time.time()
    worker = clz()
    worker.init()
    worker.run()
    worker.send()
    worker.cleanup()
    t2 = time.time()

    logger.info("End....")
