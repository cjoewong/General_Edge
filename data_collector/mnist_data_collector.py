#!/usr/bin/env python3

from .data_collector_base import DataCollectorBase
from utils import bluetootch_utils
import numpy as np

DATA_POINTS = 4000

class MNISTDataCollector(DataCollectorBase):
    """Linear Regression Data Collector

    Currently, we directly to load data from CSV files instead of collecting
    from real sensors.

    """

    def __init__(self):
        pass

    def init(self, name, config):
        self._name = name
        self._config = config
        self._file_paths = config.get("dataFilePaths")
        self._data = None

    def cleanup(self):
        pass

    def send(self, **kwargs):
        """Send data to the downStream Pi
        """
        down_addr = kwargs.get('down_addr')
        bluetootch_utils.sendData(self._data, down_addr, 1)

    def run(self, **kwargs):
#       raw_data = []
        mnist = np.load(self._file_paths[0])[:DATA_POINTS]
        print('Collect data: {0}'.format(len(mnist)))
        self._data = {"from_pi": self._name, "data": mnist}
