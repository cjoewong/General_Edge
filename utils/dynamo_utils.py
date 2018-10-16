"""
Centralized location for handling all requests to Amazon's DynamoDB

@ Original Author   : Chege Gitau

"""
#______________________________________________________________________________

import boto3
import os
from botocore.exceptions import ClientError
import logging
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id = os.environ['aws_access_key_id'],
    aws_secret_access_key = os.environ['aws_secret_access_key'],
    region_name = 'us-east-1'
)
client = boto3.client(
    "dynamodb",
    aws_access_key_id=os.environ['aws_access_key_id'],
    aws_secret_access_key=os.environ['aws_secret_access_key'],
    region_name='us-east-1'
)

#_______________________________________________________________________________



class Table:

    def __init__(self, nameOfTable, hashKey = None, rangeKey = None, readCapUnits = 2, writeCapUnits = 2):
        """
        Initializes an instance of this class bearing a reference to a table.

        Param(s):
            (String)    Name of the table

        Optional Param(s):
            hashKey (list)      Name of the hash key and either 'S', 'N' or 'B' to specify type.
            rangeKey (list)     Name of the range key and either 'S', 'N' or 'B' to specify type.
            readCapUnits (int)  Read capacity units. 2 if not specified.
            writeCapUnits (int) Write capacity units. 2 if not specified.
        """
        self.logger = logging.getLogger()
        self.nameOfTable = nameOfTable
        print(nameOfTable)
        # If the table doesn't exist, create one.
        if (not self.tableExists(nameOfTable)):
            self.table = self.createTable(nameOfTable, hashKey, rangeKey, readCapUnits, writeCapUnits)
        # Else return a table resource corresponding to an existing table
        else:
            self.table = dynamodb.Table(nameOfTable)
            self.logger.info("Fetched the table %s created at %s", self.table.table_name, self.table.creation_date_time)

    def createTable(self, nameOfTable, hashKey, rangeKey, readCapUnits = 2, writeCapUnits = 2):
        """
        Param(s):
            (String)            Name of the table that you wish to create.
            (list)              Name of the hash key and either 'S', 'N' or 'B' to specify type.
            (list)              Name of the range key and either 'S', 'N' or 'B' to specify type.

        Optional Param(s):
            readCapUnits (int)  Read capacity units. 2 if not specified.
            writeCapUnits (int) Write capacity units. 2 if not specified.

        Returns a table resource that corresponds to an active table on DynamoDB
        Raises a ReferenceError is no KeySchema values are provided.

        """
        # Raise exceptions for invalid requests
        if hashKey == None and rangeKey == None:
            tip = "Try an initialization such as Table('nameOfTable', ['last_name', 'S'])"
            self.log(tip)
            raise ReferenceError("New tables need at least a hash key or a range key.")

        # Create the DynamoDB table
        keySchema = []
        attributeDefs = []
        if hashKey != None:
            keySchema.append({
                'AttributeName' : hashKey[0],
                'KeyType'       : 'HASH'
            })
            attributeDefs.append({
                'AttributeName' : hashKey[0],
                'AttributeType' : hashKey[1]
            })

        if rangeKey != None:
            keySchema.append({
                'AttributeName' : rangeKey[0],
                'KeyType'       : 'RANGE'
            })
            attributeDefs.append({
                'AttributeName' : rangeKey[0],
                'AttributeType' : rangeKey[1]
            })

        table = dynamodb.create_table(
            TableName = nameOfTable,
            KeySchema = keySchema,
            AttributeDefinitions = attributeDefs,
            ProvisionedThroughput = {
                'ReadCapacityUnits' : readCapUnits,
                'WriteCapacityUnits': writeCapUnits
            }
        )

        # Wait for the table to exist since creation is asynchronous
        table.meta.client.get_waiter('table_exists').wait(TableName = nameOfTable)

        # Return a reference to the table
        self.logger.info("Table '%s' created at %s", table.table_name, table.creation_date_time)
        return table

    def tableExists(self, nameOfTable):
        """
        Param(s)
            (String) Name of a (potential) DynamoDB table

        Returns True if such a table actually exists, False otherwise.
        """
        try:
            return dynamodb.Table(nameOfTable).table_status == "ACTIVE"
        except:
            return False

    def getAttributes(self):
        """
        Returns a string that shows the atribute definitions for the table's
        key schema.

        """
        attributeMap = {
            'S' : '(type = String)',
            'B' : '(type = Binary)',
            'N' : '(type = Number)'
        }
        attributes = self.table.attribute_definitions
        listOfAttr = []
        for attr in attributes:
            listOfAttr.append("'" + attr['AttributeName'] + "' " + attributeMap[attr['AttributeType']])
        return ' and '.join(listOfAttr)

    def addItem(self, itemData):
        """
        Param(s):
            (dict) Specifies the data that you wish to add to the table.

        Returns a dict containing metrics and stats from AWS.
        Overwrites old values if the KeySchema values are already in present.
        Raises a ClientError exception if the input dict doesn't contain mandatory keys.
        """
        try:
            print(itemData)
            return self.table.put_item(Item = itemData)
        except ClientError as e:
            tip = "Your dict must have the following keys : " + self.getAttributes()
            self.log(tip)
            raise(e)

    def getItem(self, itemKey):
        """
        Param(s):
            (dict) The key-value pair(s) that you wish to search for in the table.

        Returns a dict containing the matching item; empty dict if no match found.
        Raises a ClientError exception if the input dict's keys don't match the
        stored attributes.

        """
        try:
            return self.table.get_item(Key = itemKey)['Item']
        except KeyError:
            return {}
        except ClientError as e:
            tip = "The correct keys are " + self.getAttributes()
            self.log(tip)
            raise(e)

    def getAllItems(self):
        """
        Scan all the items in this table. Return a paginator iterable.

        """
        paginator = client.get_paginator("scan")
        return paginator.paginate(TableName=self.nameOfTable)

    def compareValues(self, itemKey, keyToLookUp, expectedValue, testForEquality):
        """
        Helper method for checking values in a DynamoDB table.
        Useful for setting and reading tables that act as triggers

        Param(s):
            (dict)  The key-value pair(s) that you wish to search for in the table.
            (?)     The key for the value that you wish to check
            (?)     The expected value that's going to be tested against.
            (bool)  If True, method returns True iff the expectedValue matches the value in the table

        Returns a boolean and a copy of the data found at the key of interest.

        """
        item = self.getItem(itemKey)
        booleanResult = expectedValue == item[keyToLookUp]
        if testForEquality:
            return booleanResult, item[keyToLookUp]
        else:
            return not booleanResult, item[keyToLookUp]

    def log(self, tip):
        """
        Param(s):
            (String) Tip for handling the exception that you're about to raise.

        Logs message to the console. Useful for providing tips to the user.
        """
        print("---------------------------")
        print("Pro-Tip :", tip)
        print("---------------------------")
#_______________________________________________________________________________
