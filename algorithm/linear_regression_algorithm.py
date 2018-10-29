#!/usr/bin/env python3

from .algorithm_base import AlgorithmBase
from utils.dynamo_utils import Table
from decimal import Decimal
import numpy as np
import time
import logging


class LinearRegression(AlgorithmBase):

    def __init__(self):
        pass

    def init(self, name, config):
        """
        Init the required arguments we will use in the trainning stage.

        Param(s):
            name    The name of current Pi
            config  The yaml configuration object passed from pi_manager

        """
        self._name = name
        self._config = config
        self._lr = 0.0001
        self._lambd = 0.1
        self._epochs = 1
        self._local = None
        self._down_stream_data = None
        self._logger = logging.getLogger('')
        self._logger.info('LinearRegression Init finished...')
        self.process_time = 0

    def run(self, **kwargs):
        """
        Start to train LinearRegression Model

        """
        self._logger.info('LinearRegression train start...')
        start_time = time.time()
        print(self._config.get("local", True))
        self._local = self._config.get("local", True)
        train_data = kwargs.get("train_data", [])
        X, y = self.get_data(train_data)
        if not self._local:
            self._down_stream_data = {'x': X, 'y': y}
            self._logger.info('LinearRegression skip train...')
            return

        X = np.array(X)
        y = np.array(y)
        w, b = self.gradient_descent(X, y)
        self._down_stream_data = {'w': w, 'b': b}
        end_time = time.time()
        self.process_time = end_time - start_time
        self._logger.info('LinearRegression train end...')

    def get_data(self, train_data):
        """
        Parse the tarinning data from the DataCollector

        Param(s):
            train_data The data transfered from DataCollector
        """
        X = []
        y = []
        for data in train_data:
            for row in data.get("data"):
                X.append(list(map(lambda x : float(x), row[:-1])))
                y.append(float(row[-1]))
        return X, y

    def cleanup(self):
        """
        Clean up
        """
        pass

    def gradient_descent(self, design_matrix, target_matrix):
        """
        Gradient descent main function

        Param(s):
            design_matrix   The initial data matrix
            target_matrix   The true data matrix
        """
        target_matrix = target_matrix[:, None]
        count = 0
        w_old = np.zeros((3, 1))
        w_new = np.zeros((3, 1))
        E_old = 0
        E_new = 0
        delta_E = np.zeros((len(design_matrix), 3))
        learning_rate = 0.001

        # tolerance = 1e-5
        while True:
            w_old = w_new

            for i in range(len(design_matrix)):
                delta_E[i, :] = delta_E[i,:] + (target_matrix[i] - np.dot(np.matrix(design_matrix[i, :]), np.matrix(w_old))) * design_matrix[i,:]


            w_new = w_old + learning_rate * np.matrix(delta_E[i, :] / (len(design_matrix))).T
            E_old = E_new

            for i in range(len(design_matrix)):
                E_new = E_new + (target_matrix[i]- np.dot(np.matrix(design_matrix[i, :]), np.matrix(w_new))) ** 2
                E_new = E_new / 2

            if E_new > E_old:
                learning_rate = learning_rate / 2

            count = count + 1
            if count % 20 == 0:
                print(str(count), " iterations so far...")

            # Test if restricting iterations affects the quality
            if count == 50:
                break

        return w_new, 0

    def send(self, **kwargs):
        """
        Send the model to the downstream and here it is Dynamo
        """

        bt_time = kwargs.get('bt_time')
        self._logger.info('LinearRegression send start...')
        if self._local is None:
            raise RuntimeError('Please train model first')

        start_time = time.time()
        table = Table(self._config.get('downStream'))
        room = self._config.get('room')
        sensor = self._config.get('sensor')

        # Prepare the upload payload
        item = table.getItem({
            'forum'     : room,
            'subject'   : sensor
        })

        if not self._local:
            X = self._down_stream_data.get('x')
            y = self._down_stream_data.get('y')

            num_of_rows = len(X)
            aggregated_items = []

            for i in range(num_of_rows):
                currentItem = {}
                currentItem['X_1'] = Decimal(str(X[i][0]))    # Time
                currentItem['X_2'] = Decimal(str(X[i][1]))    # Pressure
                currentItem['X_3'] = Decimal(str(X[i][2]))    # Humidity
                currentItem['Y'] = Decimal(str(y[i]))         # Temperature
                aggregated_items.append(currentItem)
            item['aggregated_data'] = aggregated_items
            processed_data_size = -1
        else:
            w = self._down_stream_data.get('w')
            b = self._down_stream_data.get('b')
            item['feature_A'] = Decimal(str(float(w[0])))
            item['feature_B'] = Decimal(str(float(w[1])))
            item['feature_C'] = Decimal(str(float(w[2])))
            processed_data_size = w.nbytes

        item['bt_time'] = Decimal(str(bt_time))
        item['processed_data_size'] = Decimal(str(processed_data_size))
        item['calculation_time'] = Decimal(str(self.process_time))
        table.addItem(item)
        end_time = time.time()

        upload_time = end_time - start_time
        item['upload_time'] = Decimal(str(upload_time))
        item['timestamp'] = Decimal(str(end_time))
        table.addItem(item)
        item.pop('aggregated_data', None)
        item.pop('forum', None)
        item.pop('subject', None)

        self._logger.info('Data sent to Dynamo...')
