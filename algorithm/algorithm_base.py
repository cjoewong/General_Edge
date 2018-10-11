#!/usr/bin/python3

class AlgorithmBase:
    """
    Interface for the algorithm part
    """

    def __init__(self):
        pass

    def init(self, name, config):
        raise RuntimeError("Need to be implemented!")

    def cleanup(self):
        raise RuntimeError("Need to be implemented!")

    def run(self):
        raise RuntimeError("Need to be implemented!")

    def send(self, down_addr):
        raise RuntimeError("Need to be implemented!")
