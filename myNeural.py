import math
from random import randint


# features being used for prediction
feature_one = 2
feature_two = 1

# weights
connection_one = randint(0,10)
connection_two = randint(0,10)
bias = randint(0,10)


def propagation(f1,f2,w1,w2,b):
	"""
	Propagation function:
		Returns weight which is the sum of the connections/weights multiplied by the features. 
		Bias is added to serve as threshold to shift activation function (wiki)

	"""
	return (f1 * w1 + f2 * w2) + b



def sigmoid(weight):
	"""
	Sigmoid/logistic function:
		Compresses weight into a value between -1 and 1
		Returns the prediction for the neuron.
	"""
	return 1/(1 + math.exp(-weight))


def NeuralNet(f1,f2,w1,w2,b):
	"""
	Sigmoid + propagation function
		Takes features, connections and bias as input
		returns prediction
	"""
	weight = propagation(f1,f2,w1,w2,b)
	prediction = sigmoid(weight)
	return prediction

