import numpy as np
from AlgorithmBase import AlgorithmBase

class LinearRegression(AlgorithmBase):

    def __init__(self, config):
        self._config = config
        self.lr = 0.01
        self.lambd = 0.1
        self.epochs = 50
        self.w = None
        self.b = None

    def run(self, X, y, local=True):
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
            pred = X@self.w + self.b
            dloss = pred - y
            dw = X.T@dloss/X.shape[0]
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
