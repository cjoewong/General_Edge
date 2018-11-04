from .algorithm_base import AlgorithmBase
from utils.dynamo_utils import Table
from decimal import Decimal
import numpy as np
import time
import logging
import pickle

class Sigmoid:
    def forward(self, x):
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):
        return self.state * (1.0 - self.state)

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1.0 - self.state**2

    def __call__(self, x):
        return self.forward(x)

def one_hot(labels):
    res = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        res[i,labels[i]] = 1
    return res

def init_weight(d0, d1):
    std = np.sqrt(2.0 / (d0 + d1))
    return np.random.normal(0, std, (d0, d1))

class MNISTNetwork(AlgorithmBase):

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
        self._logger.info('MNIST network Init finished...')
        self.process_time = 0

    def run(self, **kwargs):
        self._logger.info('MNIST network train start...')
        start_time = time.time()
        print(self._config.get("local", True))
        self._local = self._config.get("local", True)
        train_data = kwargs.get("train_data", [])
        print(train_data)
        X, y = self.get_data(train_data[0].get("data"))
        if not self._local:
            self._down_stream_data = {'x': X, 'y': y}
            self._logger.info('MNIST network skip train...')
            return

        W = self.neural_network(X, y)
        self._down_stream_data = {'w': W}
        end_time = time.time()
        self.process_time = end_time - start_time
        self._logger.info('MNIST network train end...')

    def get_data(self, train_data):
        X = train_data[:,:-1]
        y = train_data[:,-1]
        return X, y

    def cleanup(self):
        pass

    def neural_network(self, images, labels):
        init_lr = 0.01
        epoch = 10
        batch_size = 100
        H = 200
        D = 28
        O = 10
        indexes = np.arange(labels.shape[0])
        W1 = init_weight(H, D*D+1)
        W2 = init_weight(O, H+1)
        for e in range(epoch):
            print("start epoch {0}".format(e+1))
            pos = 0
            lr = init_lr/(e//10 + 1)
            np.random.shuffle(indexes)
            while pos < indexes.shape[0]:
                batch_data = images[indexes[pos:pos+batch_size]]
                batch_label = labels[indexes[pos:pos+batch_size]]
                pos += batch_size
                data = np.reshape(batch_data, (-1, D*D))
                b1 = np.ones((data.shape[0], 1))
                b2 = np.ones((data.shape[0], 1))
                data = np.concatenate((data, b1), axis=1)
                activation1 = Tanh()
                activation2 = Sigmoid()
                # Initialize weight as uniform distribution.
                h1 = activation1(np.dot(data, W1.T))
                h1 = np.concatenate((h1, b2), axis=1)
                z = activation2(np.dot(h1, W2.T))
                pred = np.argmax(z, axis=1)[:,None]
                oh_label = one_hot(batch_label)
                L = oh_label*np.log(z)+(1-oh_label)*np.log(1-z)
                dL = -(oh_label/z - (1-oh_label)/(1-z))*activation2.derivative()
                dy = np.dot(dL, W2)
                du = dy[:,:-1]*activation1.derivative()
                dW1 = 1/batch_label.shape[0]*np.dot(du.T, data)
                dW2 = 1/batch_label.shape[0]*np.dot(dL.T, h1)
                W1 -= lr*dW1
                W2 -= lr*dW2
        W = [W1, W2]
        return W

    def send(self, **kwargs):
#       down_addr = kwargs.get('down_addr')
        bt_time = kwargs.get('bt_time')
        self._logger.info('MNIST neural network send start...')
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

            # Numpy indexes follow the [row][column] convention
            # ndarray.shape returns the dimensions as a (#OfRows, #OfColumns)
            # Both of our matrices have the same number of rows, hence one measure is enough
            item['image'] = pickle.dumps(X)
            item['Y'] = pickle.dumps(y)
        else:
            w = self._down_stream_data.get('w')
            w0 = pickle.dumps(w[0])
            w1 = pickle.dumps(w[1])
            pickle.dump(w[0], open('mnistA_dump', 'wb'))
            print(len(w0))
            item['w1'] = w0
            item['w2'] = w1

        item['bt_time'] = Decimal(str(bt_time))
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
