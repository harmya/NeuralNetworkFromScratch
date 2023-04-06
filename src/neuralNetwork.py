import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# Sigmoid function for activation
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

# Neural Network class

class NeuralNetwork(object):

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        
        return a





# Test the neural network
neuralNetwork = NeuralNetwork([2, 3, 1])

n_test = 10
test_data = [(np.random.randn(2, 1), np.random.randn(1, 1)) for i in range(n_test)]

for x, y in test_data:
    a_input = x
    a_output = neuralNetwork.feedforward(a_input)
    cost = np.sum((a_output - y)**2)
    print("Input: ", a_input)
    print("Output: ", a_output)
    print("Expected: ", y)
    print("Cost: ", cost)
    print("")
    

        
