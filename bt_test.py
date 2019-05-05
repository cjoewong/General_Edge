from btpycom import *

def onStateChange(state,msg):
	print(state)

serviceName = 'bt_test'
server = BTServer(serviceName,stateChanged = onStateChange)