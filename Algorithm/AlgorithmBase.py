#!/usr/bin/python3

class AlgorithmBase:
    """
    Interface for the algorithm part
    """

    def __init__(self, config):
        self._config = config

    def init(self):
        pass

    def cleanup(self):
        pass

    def run(self):
        pass

    def upload(self):
        pass
