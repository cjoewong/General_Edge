import numpy as np
import time
from .algorithm_base import AlgorithmBase


class LinearRegression(AlgorithmBase):

    def __init__(self):
        pass

    def init(self, name, config):
        self._name = name
        self._config = config
        self.lr = 0.01
        self.lambd = 0.1
        self.epochs = 50
        self.w = None
        self.b = None

    def run(self, **kwargs):
        local = kwargs.get("local", True)
        train_data = kwargs.get("train_data", [])
        X = []
        y = []
        for data in train_data:
            for row in data.get("data"):
                X.append(list(map(lambda x : float(x), row[:-1])))
                y.append(float(row[-1]))

        X = np.array(X)
        y = np.array(y)

        print(X)
        if local:
            return self.gradient_descent(X, y)
        else:
            return self.transmit(X, y)

    def cleanup(self):
        self.w = None
        self.b = None

    def gradient_descent(self, X, y):
        if self.w == None:
            self.w = self.init_w(X.shape[1], 1)
        if self.b == None:
            self.b = 0

        for epoch in range(self.epochs):
            pred = X.dot(self.w) + self.b
            dloss = pred - y
            dw = np.dot(X.T,dloss)/X.shape[0]
            db = np.sum(dloss)/X.shape[0]
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self.w,self.b


    def transmit(self, X, y):
        pass

    def init_w(self, dim0, dim1):
        return np.zeros((dim0, dim1))

    def init_b(self, dim0):
        return np.zeros((dim0, 1))

    def send(self, down_addr):
        print("TO BE IMPLEMENTED")
        time.sleep(3)
        print("TO BE IMPLEMENTED")
        time.sleep(3)
        print("TO BE IMPLEMENTED")
