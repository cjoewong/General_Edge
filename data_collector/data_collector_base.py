#!/usr/bin/env python3

from abc import ABCMeta, abstractmethod


class DataCollectorBase(metaClass=ABCMeta):
    """

    """

    def __init__(self, config):
        self._config = config
        pass

    def init(self):
        pass

    def cleanup(self):
        pass

    @abstractmethod
    def run(self):
        pass
