from .algorithm_base import AlgorithmBase
from utils.dynamo_utils import Table
from utils.s3_client import S3Client
from decimal import Decimal
import numpy as np
import time
import logging
import pickle
import csv

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

def softmax(x):
    expx = np.exp(x)
    return expx / np.sum(expx, axis=1)[:,None]

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
        self.s3client = S3Client()

    def run(self, **kwargs):
        self._logger.info('MNIST network train start...')
        start_time = time.time()
        #print(self._config.get("local", True))
        self._local = self._config.get("local", True)
        train_data = kwargs.get("train_data", [])
        weights = kwargs.get("weights", None)
        iterations = kwargs.get("iterations", 20)
        batch_size = kwargs.get("batch_size", 100)
		
        X, y = self.get_data(train_data[0].get("data"))
        if not self._local:
            self._down_stream_data = {'x': X, 'y': y}
            self._logger.info('MNIST network skip train...')
            return

        W = self.neural_network(X, y, iterations, batch_size, weights)
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

    def neural_network(self, images, labels, iterations, batch_size,W=None):
        init_lr = 0.01
        epoch = iterations
        batch_size = batch_size
        D  = 28
        L1 = 32
        O  = 10
        file_name = 'accuracies_'+str(epoch)+'_'+str(batch_size)+'.csv'
        indexes = np.arange(labels.shape[0])
        if(W is not None):
            W  = init_weight(D*D+1, L1)
            W1 = init_weight(L1, O)
            print("Shape of W1 : ",np.shape(W1))
        else:
            W = W.reshape(D*D+1,L1)
            W1 = W1.reshape(L1,O)
        accuracies = []
        for e in range(epoch):
            #print("start epoch {0}".format(e+1))
            pos = 0
            lr = init_lr/(e//10 + 1)
            np.random.shuffle(indexes)
            while pos < indexes.shape[0]:
                #print("Shape of W : ",np.shape(W))
                #print("Shape of W1: ",np.shape(W1))
                batch_data = images[indexes[pos:pos+batch_size]]
                batch_label = labels[indexes[pos:pos+batch_size]]
                pos += batch_size
                b1 = np.ones((batch_data.shape[0], 1))
                data = np.concatenate((batch_data, b1), axis=1)
                activation = Sigmoid()
                activation_2 = Sigmoid()
                # Initialize weight as uniform distribution.
                z_l0 = activation(np.dot(data, W))
                #print("Shape of z_l0 : ",np.shape(z_l0))				
                z_l1 = softmax(activation_2(np.dot(z_l0, W1)))
                #print("Shape of z_l1 : ",np.shape(z_l1))
                pred = np.argmax(z_l1, axis=1)[:,None]
                oh_label = one_hot(batch_label)
                L = oh_label*np.log(z_l1)+(1-oh_label)*np.log(1-z_l1)
                #dL = -(oh_label/z - (1-oh_label)/(1-z))*activation.derivative()
				
                predictions = []
                for single_op in z_l1:
                    result = np.where(single_op == np.amax(single_op))[0]
                    predictions.append(result[0])
                accuracies.append((np.sum(np.equal(predictions,batch_label))/len(predictions))*100)
				
                line = accuracies
						
                dL1 = (z_l1 - oh_label)*activation_2.derivative()
                #print("Shape of dL1 : ",np.shape(dL1))				
                dW1 = np.dot(z_l0.T, dL1)/batch_label.shape[0]
                #print("Shape of dW1 : ",np.shape(dW1))				
                dL = np.dot(W1,dL1.T).T*activation.derivative()
                #print("Shape of dL : ",np.shape(dL))
                dW = np.dot(data.T,dL)/batch_label.shape[0]
                #print("Shape of dW : ",np.shape(dW))
                
                W1 -= lr*dW1
                W -= lr*dW
                print("Loss = ",np.mean(np.abs(L)))
                #print()
        with open(file_name, 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(accuracies)
            writeFile.close()
        return W

    def send(self, **kwargs):
#       down_addr = kwargs.get('down_addr')
        bt_time = kwargs.get('bt_time')
        self._logger.info('MNIST neural network send start...')
        if self._local is None:
            raise RuntimeError('Please train model first')

        start_time = time.time()
        table = Table(self._config.get('downStream'))
        #print('Downstream Table : ',table)
        room = self._config.get('room')
        #print('Downstream Room : ',room)
        sensor = self._config.get('sensor')
        #print('Downstream Sensor : ',sensor)

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
            np.save(sensor+'X.npy', X)
            np.save(sensor+'y.npy', y)
            self.s3client.upload(sensor+'X.npy', 'mnist-nerual-network', sensor+'X.npy')
            self.s3client.upload(sensor+'y.npy', 'mnist-nerual-network', sensor+'y.npy')

        else:
            w = self._down_stream_data.get('w')
            item['weight'] = pickle.dumps(w,protocol=2)

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
