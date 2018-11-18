all:
	@echo "Please specify which lambda function you want to package"
	@echo "Options: mnist_nn"

mnist_nn:
	mkdir -p package
	rm -rf package/*
	cp mnist_neural_network_lambda.py package/
	cp ../utils/dynamo_utils.py package/
