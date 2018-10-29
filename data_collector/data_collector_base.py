#!/usr/bin/env python3


class DataCollectorBase():
    """
    Data collector interface
    """

    def __init__(self, config):
        self._config = config
        pass

    def init(self):
        pass

    def cleanup(self):
        pass

    def send(self):
        pass

    def run(self):
        pass
