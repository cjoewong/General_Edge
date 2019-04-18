#!/usr/bin/env python3

from data_collector_base import DataCollectorBase


class LinearRegressionDataCollector(DataCollectorBase):
    """Linear Regression Data Collector

    Currently, we directly to load data from CSV files instead of collecting
    from real sensors.

    """

    def init(self, name, config):
        """
        Init the required arguments we will use in the data collector stage.

        Param(s):
            name    The name of current Pi
            config  The yaml configuration object passed from pi_manager
        """
        self._name = name
        self._config = config
        self._file_paths = config.get("dataFilePaths")
        self._data = None

    def cleanup(self):
        """
        Clean up
        """
        pass

    def send(self, **kwargs):
        """
        Send data to the downStream Gateway Pi
        """
        down_addr = kwargs.get('down_addr')
        return self._data

    def run(self, **kwargs):
        """
        Collect data from given csv file
        """
        raw_data = []
        for p in self._file_paths:
            with open(p, 'r') as f:
                for line in f:
                    attr = [0]
                    attr.extend(line.strip().split(',')[1:])
                    if len(attr) != 4:
                        continue
                    raw_data.append(attr)

        print('Collect data: {0}'.format(len(raw_data)))
        self._data = {"from_pi": self._name, "data": raw_data}
