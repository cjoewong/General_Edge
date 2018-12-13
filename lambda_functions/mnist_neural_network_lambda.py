"""
Mnist Neural Network Lambda side prrediction function.

"""

import numpy as np
import time
import decimal
import boto3
import pickle

from dynamo_utils import Table

s3_client = boto3.client('s3')
BUCKET_NAME = 'mnist-nerual-network'
SMALL_TEST_FILE = 'mnist_C_small.npy'
FULL_TEST_FILE = 'mnist_C.npy'
TRAIN_DATA_A = 'mnist_A.npy'
TRAIN_DATA_B = 'mnist_B.npy'

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

def neural_network(images, labels):
    init_lr = 0.01
    epoch = 20
    batch_size = 100
    D = 28
    O = 10
    indexes = np.arange(labels.shape[0])
    W = init_weight(D*D+1, O)
    for e in range(epoch):
        print("start epoch {0}".format(e+1))
        pos = 0
        lr = init_lr/(e//10 + 1)
        np.random.shuffle(indexes)
        while pos < indexes.shape[0]:
            batch_data = images[indexes[pos:pos+batch_size]]
            batch_label = labels[indexes[pos:pos+batch_size]]
            pos += batch_size
            b1 = np.ones((batch_data.shape[0], 1))
            data = np.concatenate((batch_data, b1), axis=1)
            activation = Sigmoid()
            # Initialize weight as uniform distribution.
            z = softmax(activation(np.dot(data, W)))
            pred = np.argmax(z, axis=1)[:,None]
            oh_label = one_hot(batch_label)
            L = oh_label*np.log(z)+(1-oh_label)*np.log(1-z)
            #dL = -(oh_label/z - (1-oh_label)/(1-z))*activation.derivative()
            dL = (z - oh_label)*activation.derivative()
            dW = 1/batch_label.shape[0]*np.dot(data.T, dL)
            W -= lr*dW
    return W

def lambda_handler(event, context):
    lambda_start = time.time()

    # Load test data
    download_path = '/tmp/' + FULL_TEST_FILE
    s3_client.download_file(BUCKET_NAME, FULL_TEST_FILE, download_path)
    test_data = np.load(download_path)
    X = test_data[:, :-1]
    y = test_data[:, -1]
    b1 = np.ones((X.shape[0], 1))
    X = np.concatenate((X, b1), axis=1)

    # 1. try use different trained weights
    # 2. try different combination of weight_A and weight_B
    for train_size in range(1000, 21000, 1000):
        subject = 'mnist' + str(train_size)
        # Fetch the local calculated weight from DynamoDB
        table_A = Table('mnist_A')
        item_key = {'forum': 'roomA', 'subject': subject}
        item_A = table_A.getItem(item_key)
        if 'weight' not in item_A:
            print('weight not found in mnist_A, terminate here!')
            break
        weight_A = pickle.loads(item_A['weight'].value)

        table_B = Table('mnist_B')
        item_key = {'forum': 'roomB', 'subject': subject}
        item_B = table_B.getItem(item_key)
        weight_B = pickle.loads(item_B['weight'].value)
        print("Get weight for trainning size: {0}".format(train_size))

        results = {}
        for k in np.arange(0.3, 0.8, 0.1):
            round_start = time.time()
            weight = k * weight_A + (1 - k) * weight_B
            # Prediction
            y_pred = np.dot(X, weight)
            acc = (y_pred.argmax(axis=1) == y).mean()
            print("k = {0} and acc = {1}".format(k, acc))
            round_end = time.time()
            results[str(k)] = {'acc': decimal.Decimal(str(acc)),
                               'time': decimal.Decimal(str(round_end - round_start))}
        table = Table('mnresult', ['environment', 'S'])
        result_data = {
            'environment': 'roomA_' + subject,
            'sensor': 'test_' + subject,
            'round': results,
            'time': decimal.Decimal(str(time.time() - lambda_start))
        }
        item = table.addItem(result_data)

    print("Finished")

def train_in_lambda(event, context):

    # Load test data
    download_path = '/tmp/' + FULL_TEST_FILE
    s3_client.download_file(BUCKET_NAME, FULL_TEST_FILE, download_path)
    test_data = np.load(download_path)
    X = test_data[:, :-1]
    y = test_data[:, -1]
    b1 = np.ones((X.shape[0], 1))
    X = np.concatenate((X, b1), axis=1)

    # Load train data
    download_path = '/tmp/' + TRAIN_DATA_A
    s3_client.download_file(BUCKET_NAME, TRAIN_DATA_A, download_path)
    download_path = '/tmp/' + TRAIN_DATA_B
    s3_client.download_file(BUCKET_NAME, TRAIN_DATA_B, download_path)
    train_data_A = np.load(download_path)
    train_data_B = np.load(download_path)
    train_A_X = train_data_A[:, :-1]
    train_A_y = train_data_A[:, -1]
    train_A_b1 = np.ones((train_A_X.shape[0], 1))
    #train_A_X = np.concatenate((train_A_X, train_A_b1), axis=1)
    train_B_X = train_data_B[:, :-1]
    train_B_y = train_data_B[:, -1]
    train_B_b1 = np.ones((train_B_X.shape[0], 1))
    #train_B_X = np.concatenate((train_B_X, train_B_b1), axis=1)

    results = {}
    for train_size in range(1000, 21000, 1000):
        round_start = time.time()

        print('train_size: ', train_size)
        print(train_A_X.shape)
        print(train_A_y.shape)
        images = np.concatenate((train_A_X[:train_size], train_B_X[:train_size]))
        labels = np.concatenate((train_A_y[:train_size], train_B_y[:train_size]))
        W = neural_network(images, labels)

        # test
        y_pred = np.dot(X, W)
        acc = (y_pred.argmax(axis=1) == y).mean()
        print("tain_size = {0} and acc = {1}".format(train_size, acc))
        round_end = time.time()
        results[str(train_size)] = {'acc': decimal.Decimal(str(acc)),
                           'time': decimal.Decimal(str(round_end - round_start))}
        table = Table('mnresult-lambda', ['environment', 'S'])
        subject = 'mnist_' + str(train_size)
        result_data = {
            'environment': 'roomA_' + subject,
            'sensor': 'test_' + subject,
            'round': results,
        }
        item = table.addItem(result_data)
    print('finished')
