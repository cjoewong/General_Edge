#!/usr/bin/env python3

import logging


class DependencyHandler(object):
    """
    The handler is used to resolve the dependecies between sensor and gateway
    """
    def __init__(self, global_cfg):
        """
        Init function

        Param(s):
            global_cfg  The yaml configuration object
        """
        self._global_cfg = global_cfg
        self._dependencies = {}
        self._logger = logging.getLogger()

        self._parse()

    def _parse(self):
        """
        Analyze the global dependencies
        """
        for name, info in self._global_cfg.items():
            down_stream = info.get("downStream")
            if down_stream is not None and down_stream != name:
                prerequisites = self._dependencies.get(down_stream, set())
                prerequisites.add(name)
                self._dependencies[down_stream] = prerequisites

    def add_resolved_dependency(self, piA, piB):
        """
        Add one resolved dependency, piA -> piB

        Param(s):
            piA   The frontend pi(e.g. sensorPi)
            piB   The backend pi(e.g. gatewayPi)
        """
        try:
            prerequisites = self._dependencies.get(piB)
            print(prerequisites)
            print(piA)
            prerequisites.remove(piA)
        except KeyError:
            self._logger.error("Invalid PiB - {0}".format(piB))
        except ValueError:
            self._logger.error("Remove a non-exist piA - {0}".format(piA))

    def dependency_resolved(self, pi_name):
        """
        Return if this pi's dependencies are all resolved

        Param(s):
            pi_name  The pi you want to check

        Return
            true/false Indicate if the dependencies resolved or not
        """
        return self._dependencies.get(pi_name) is None or len(self._dependencies.get(pi_name)) == 0
