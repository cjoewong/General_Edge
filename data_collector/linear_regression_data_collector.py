#!/usr/bin/env python3

from .data_collector_base import DataCollectorBase


class LinearRegressionDataCollector(DataCollectorBase):
    """Linear Regression Data Collector

    Currently, we directly to load data from CSV files instead of collecting
    from real sensors.

    """

    def __init__(self):
        pass

    def init(self, config):
        self._file_paths = config.get("dataFilePaths")

    def cleanup(self):
        pass

    def send(self):
        pass

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
        return raw_data
