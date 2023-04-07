import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


# Sigmoid function for activation
def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

# Derivative of the cost function with respect to the activation of the output layer
def cost_derivative_wrt_a(output_activations, y):
    return (output_activations-y)


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

    def backpropogation(self, x, y):
        # initialize the nabla cost vectors
        nabla_cost_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_cost_biases = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        # x is the activation of the input layer
        activation = x
        activations = [x]

        z_vectors = []

        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation) + bias
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # now we have the activations of all the layers and their respective z vectors with their biases
        # so we can do the backpropogation

        # calculate the delta of the output layer

        derivative_cx_wrt_bias = cost_derivative_wrt_a(activations[-1], y) * sigmoid_derivative(z_vectors[-1])
        nabla_cost_biases[-1] = derivative_cx_wrt_bias

        # derivative of cost function with respect to the weights of the output layer also depends on the activation of the previous layer
        nabla_cost_weights[-1] = np.dot(derivative_cx_wrt_bias, activations[-2].T)

        # now we loop backwards through each layer to calculate the nabla cost vectors for that layer

        for l in range(2, self.num_layers):
            z = z_vectors[-l]
            derivative_a_wrt_z = sigmoid_derivative(z)
            derivative_cx_wrt_bias = np.dot(self.weights[-l+1].T, derivative_cx_wrt_bias) * derivative_a_wrt_z
            nabla_cost_biases[-l] = derivative_cx_wrt_bias
            nabla_cost_weights[-l] = np.dot(derivative_cx_wrt_bias, activations[-l-1].T)
        
        return (nabla_cost_biases, nabla_cost_weights)
        

        







# Test the neural network



    

        
