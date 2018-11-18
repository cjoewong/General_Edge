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
        item_key = {'forum': 'roomB', 'subject': 'sensorB'}
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
            'environment': 'roomA',
            'sensor': 'test_' + subject,
            'round': results,
            'time': decimal.Decimal(str(time.time() - lambda_start))
        }
        item = table.addItem(result_data)

    print("Finished")
