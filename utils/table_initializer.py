'''
Initializes all the tables that will be used by the experiment

@ Original Author   : Chege Gitau

'''
#_______________________________________________________________________________

from dynamo_utils import Table

#_______________________________________________________________________________

def initializeTable(nameOfTable, hashKey, rangeKey, hashVal, rangeVal):
    '''
    Helper method for initializing tables. Assumes all inputs are strings

    Param(s):
        (String)   Name of the table being created
        (String)   Name of the hash key. Assumed to be AWS type 'S'
        (String)   Name of the range key. Assumed to be AWS type 'S'
        (String)   Value of the hash key for the first item
        (String)   Value of the range key for the first item

    '''
    table = Table(nameOfTable, [hashKey, 'S'], [rangeKey, 'S'])
    table.addItem({
        hashKey    : hashVal,
        rangeKey   : rangeVal
    })

#_______________________________________________________________________________

# Initialize all the required tables
# remote data
initializeTable('mnist_A', 'forum', 'subject', 'roomA', 'mnist1000')
initializeTable('mnist_B', 'forum', 'subject', 'roomB', 'mnist1000')
initializeTable('mnist_C', 'forum', 'subject', 'roomC', 'mnist1000')
initializeTable('sensingdata_A', 'forum', 'subject', 'roomA', 'sensorA')
initializeTable('sensingdata_B', 'forum', 'subject', 'roomB', 'sensorB')
initializeTable('sensingdata_C', 'forum', 'subject', 'roomC', 'sensorC')
initializeTable('sensingdata_D', 'forum', 'subject', 'roomD', 'sensorD')
initializeTable('sensingdata_E', 'forum', 'subject', 'roomE', 'sensorE')
initializeTable('sensingdata_F', 'forum', 'subject', 'roomF', 'sensorF')
initializeTable('sensingdata_G', 'forum', 'subject', 'roomG', 'sensorG')
# feature
initializeTable('testresult', 'environment', 'sensor', 'roomA', 'sensorA&B&C')
initializeTable('mnresult', 'environment', 'sensor', 'roomA_mnist_1000', 'test_mnist_1000')
