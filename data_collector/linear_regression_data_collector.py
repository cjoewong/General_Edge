#!/usr/bin/env python3

from DataCollectorBase import DataCollectorBase


class LinearRegressionDataCollector(DataCollectorBase):
    """

    """

    def __init__(self):
        pass

    def init(self, *file_paths):
        assert(len(file_paths) != 0)
        self._file_paths = [p for p in file_paths]

    def cleanup(self):
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
