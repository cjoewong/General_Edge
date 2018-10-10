#!/usr/bin/env python3

import logging


class DependencyHandler(object):
    """
    The handler is used to resolve the dependecies between sensor and gateway
    """

    def __init__(self, global_cfg):
        self._global_cfg = global_cfg
        self._dependencies = {}
        self._logger = logging.getLogger()
        # Start parser
        self._parse()

    def _parse(self):
        """Analysis the global dependencies
        """
        for name, info in self._global_cfg.items():
            down_stream = info.get("downStream")
            prerequisites = self._dependencies.get(down_stream, set())
            prerequisites.add(name)

    def add_resolved_dependency(self, piA, piB):
        """
        Add one resolved dependency, piA -> piB
        """
        try:
            prerequisites = self._dependencies.get(piB)
            prerequisites.remove(piA)
        except KeyError:
            self._logger.error("Invalid PiB - {0}".format(piB))
        except ValueError:
            self._logger.error("Remove a non-exist piA - {0}".format(piA))

    def dependency_resolved(self, pi_name):
        """
        Return if this pi's dependencies are all resolved

        @return true/false Indicate if the dependencies resolved or not
        """
        return len(self._dependencies.get(pi_name)) == 0
