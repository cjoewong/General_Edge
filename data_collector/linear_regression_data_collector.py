#!/usr/bin/env python3

from .data_collector_base import DataCollectorBase
from ..utils import bluetootch_utils

class LinearRegressionDataCollector(DataCollectorBase):
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

    def send(self, down_addr):
        """Send data to the downStream Pi
        """
        bluetootch_utils.sendData(self._data, down_addr, 1)

    def run(self):
        raw_data = []
        for p in self._file_paths:
            with open(p, 'r') as f:
                for line in f:
                    attr = [0]
                    attr.extend(line.split(',')[1:])
                    if len(attr) != 4:
                        continue
                    raw_data.append(attr)

        print(f'Collect data: {len(raw_data)}')
        self._data = {"from_pi": self._config}
