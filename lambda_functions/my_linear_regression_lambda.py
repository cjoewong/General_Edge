'''
A linear regression model that runs on AWS Lambda

AWS needs a zip file because it doesn't have numpy (so I can't use the console editor)
Make sure the zip file name, .py name and the handler name on Lambda coincide.

@ Original Author :	Liang Zheng
@ Modified by	  : Chege Gitau

'''

#_______________________________________________________________________________

import numpy as np
import decimal
import time

from dynamo_utils import Table

print('Loading function')


def lambda_handler(event, context):
	# Fetch the DynamoDB resource
	tStart = time.time()

	# Change: Getting the number of samples from the 'SampleSize' table is tricky
	# When we'll have multiple Pi's, keeping track of this number will be buggy
	# For this reason, I'm setting the value of 'datanum' to the number of items
	# that we're going to get from the table containing the aggregated sensor data

	# Initialize helper variables
	featurenum = 3
	collectornum = 4
	betam = np.zeros((featurenum,collectornum))
	dataBytesFeatures = 0
	numSensors = 0
	
	def read_data_lc_true(table_id,sensor_id,features=betam,local_compute=True):

		# Fetch the features calculated by Gateway A
		table_name = 'sensingdata_'+table_id
		forum = 'room'+table_id
		subject  = 'sensor'+table_id
		table = Table(table_name)
		itemKey = {'forum' : forum, 'subject' : subject}
		item = table.getItem(itemKey)
		ts = item['timestamp']
		features[0][sensor_id] = item['feature_A']
		features[1][sensor_id] = item['feature_B']
		features[2][sensor_id] = item['feature_C']
		return features,ts
	
	def read_data_lc_false(table_id,local_compute=False):
		# Fetch the aggregated data from Gateway C
		table_name = 'sensingdata_'+table_id
		forum = 'room'+table_id
		subject  = 'sensor'+table_id
		
		table = Table(table_name)
		itemKey = {'forum' : forum, 'subject' : subject}
		item = table.getItem(itemKey)
		aggregatedData = item['aggregated_data']
		
		datanum = len(aggregatedData)
		print(datanum)
		X = np.zeros((datanum,featurenum))
		y = np.zeros((datanum,1))

		for i in range(datanum):
			X[i][0] = aggregatedData[i]['X_1']
			X[i][1] = aggregatedData[i]['X_2']
			X[i][2] = aggregatedData[i]['X_3']
			y[i][0] = aggregatedData[i]['Y']
		
		return X,y,datanum
	
	# Read the tables for every sensor
	
	#_,tsA = read_data_lc_true('A',0,)
	#_,tsB = read_data_lc_true('B',1,)
	#_,tsD = read_data_lc_true('D',2,)
	#_,tsE = read_data_lc_true('E',3,)
	#read_data_lc_true('F',4,)
	
	X,y,datanum = read_data_lc_false('C')
	print()
	
	def prox_simplex(y):
		# projection onto simplex
		n = len(y)
		print()
		print("Simplex Projection")
		print('y :',y)
		val = -np.sort(-y)
		print('val :',val)
		suppt_v = np.cumsum(val) - np.arange(1, n+1, 1) * val
		print('suppt_v : ',suppt_v)
		k_act = np.sum(suppt_v < 1)
		print('k_act : ',k_act)
		lam = (np.sum(val[0:k_act]) - 1.0) / k_act
		print('lam : ',lam)
		x = np.maximum(y-lam, 0.0)
		print('x : ',x)
		return x

	def combine(y, X, betam):
		K = betam.shape[1]
		w = np.ones((K,)) / K
		maxit = 1000
		tol = 1e-3
		Xb = np.dot(X, betam)
		step = 1.0 / np.max(np.linalg.svd(Xb, full_matrices=0, compute_uv=0)) ** 2

		for it in range(maxit):
			prev_w = np.copy(w)
			res = y - np.dot(np.matrix(Xb), np.matrix(w).T)
			grad = -np.dot(np.matrix(Xb).T, np.matrix(res))
			w -= step * np.squeeze(np.asarray(grad.T))
			w = prox_simplex(w)
			if np.linalg.norm(w - prev_w) / (1e-20 + np.linalg.norm(prev_w)) < tol:
				break

		return w

	# Combine the weights
	
	def mixed_weights(y,X,betam):
		w = combine(y, X, betam)
		print(w)
		w_temp = [decimal.Decimal(str(w[i])) for i in range(collectornum)]

		wb = np.dot(np.matrix(betam), np.matrix(w).T)
		Predict_y = np.dot(np.matrix(X), wb)
		Predict_y_array = np.squeeze(np.asarray(Predict_y))

		MSE = np.sqrt(np.sum((y-np.squeeze(np.asarray(Predict_y))) ** 2)) / datanum
		print(MSE)
		MSE_temp = decimal.Decimal(str(MSE))
		tEnd = time.time()
		Lambda_ExecTime = tEnd - tStart
		tEnd_temp = decimal.Decimal(str(tEnd))
		Lambda_ExecTime_temp = decimal.Decimal(str(Lambda_ExecTime))

		Predict_y_array = Predict_y_array.tolist()
		y = y.tolist()
		for i in range(len(Predict_y_array)):
			y[i] = decimal.Decimal(str(y[i][0]))
			Predict_y_array[i] = decimal.Decimal(str(Predict_y_array[i]))
			
	def only_weights():
		hmm = 0
	
	def only_data(X,y):
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
			w_new = self.w
			E_old = 0
			E_new = 0
			delta_E = np.zeros((len(design_matrix), 3))
			learning_rate = 0.001

			# tolerance = 1e-5
			while True:
				print(w_old)
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

		X = np.array(X)
		y = np.array(y)
        
		w_temp,b = gradient_descent(X, y)
		
		return w_temp

	table = Table('testresult')
	resultData = {
		'environment' : 'roomA',
		'sensor': 'sensorA&B&C',
		'w_1' : w_temp[0],
		'w_2' : w_temp[1],
		'Prediction' : Predict_y_array,
		'Real_Data' : y,
		'Error' : MSE_temp,
		'Lambda_ExecTime' : Lambda_ExecTime_temp,
		'Time': tEnd_temp
	}
	item = table.addItem(resultData)

	# Record this run
	#resultData.pop('environment', None)
	#resultData.pop('sensor', None)
	#resultData.pop('Prediction', None)
	#resultData.pop('Real_Data', None)
	#record = table.getItem({'environment' : 'roomA', 'sensor' : 'expResults'})
	#results = record['results']
	#results.append(resultData)
	#item = table.addItem(record)


lambda_handler(35, 46)
