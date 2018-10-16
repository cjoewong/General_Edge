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
        self._name = name
        self._config = config
        self._lr = 0.0001
        self._lambd = 0.1
        self._epochs = 1
        self._local = None
        self._down_stream_data = None
        self._logger = logging.getLogger('')
        self._logger.info('LinearRegression Init finished...')

    def run(self, **kwargs):
        self._logger.info('LinearRegression train start...')
        print(self._config.get("local", True))
        self._local = self._config.get("local", True)
        print(self._local)
        print(type(self._local))
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
        self._logger.info('LinearRegression train end...')

    def get_data(self, train_data):
        X = []
        y = []
        for data in train_data:
            for row in data.get("data"):
                X.append(list(map(lambda x : float(x), row[:-1])))
                y.append(float(row[-1]))
        return X, y

    def cleanup(self):
        pass

    def gradient_descent(self, X, y):
        w = np.zeros((X.shape[1], 1))
        b = 0

        for epoch in range(self._epochs):
            pred = X.dot(w) + b
            print(pred)
            dloss = pred - y[:, None]
            dw = np.dot(X.T, dloss)/X.shape[0]
            db = np.sum(dloss)/X.shape[0]
            w -= self._lr * dw
            b -= self._lr * db
            print(w)
        return np.around(w, decimals=4), np.around(b, decimals=4)

    def send(self, down_addr):
        self._logger.info('LinearRegression send start...')
        if self._local is None:
            raise RuntimeError('Please train model first')

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

            # Numpy indexes follow the [row][column] convention
            # ndarray.shape returns the dimensions as a (#OfRows, #OfColumns)
            # Both of our matrices have the same number of rows, hence one measure is enough
            numOfRows = len(X)
            aggregatedItems = []

            for i in range(numOfRows):
                currentItem = {}
                currentItem['X_1']     = Decimal(str(X[i][0]))    # Time
                currentItem['X_2']     = Decimal(str(X[i][1]))    # Pressure
                currentItem['X_3']     = Decimal(str(X[i][2]))    # Humidity
                currentItem['Y']       = Decimal(str(y[i]))    # Temperature
                aggregatedItems.append(currentItem)
            item['aggregated_data'] = aggregatedItems
        else:
            w = self._down_stream_data.get('w')
            b = self._down_stream_data.get('b')
            item['feature_A'] = Decimal(str(float(w[0])))
            item['feature_B'] = Decimal(str(float(w[1])))
            item['feature_C'] = Decimal(str(float(w[2])))

        table.addItem(item)

        item.pop('aggregated_data', None)
        item.pop('forum', None)
        item.pop('subject', None)

        self._logger.info('Data sent to Dynamo...')
