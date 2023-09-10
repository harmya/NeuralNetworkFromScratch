import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt



# Sigmoid function for activation
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

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
    
    # now we can apply backporpogation to update the weights and biases of the network
    # but the issue is that, for the true cost function, we need to sum over all the training examples
    # that is computaionally expensive, hence we do a stochastic gradient descent
    # stochastic gradient descent is a method of updating the weights and biases of the network by taking a small step in the direction of the gradient of the cost function
    # this small step is computed by using the backpropogation algorithm for a mini batch of training examples

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_cost_biases = [np.zeros(b.shape) for b in self.biases]
        nabla_cost_weights = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_cost_biases, delta_nabla_cost_weights = self.backpropogation(x, y)
            nabla_cost_biases = [bias + delta_bias for bias, delta_bias in zip(nabla_cost_biases, delta_nabla_cost_biases)]
            nabla_cost_weights = [weight + delta_weight for weight, delta_weight in zip(nabla_cost_weights, delta_nabla_cost_weights)]

        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_cost_weights)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_cost_biases)]
        
    # now we can train the network by using the update_mini_batch function 
    # and implementing the stochastic gradient descent algorithm

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data is not None:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    # now we can evaluate the network by using the feedforward function
    # and comparing the output of the network with the actual output

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

# Test the neural network



def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def value_from_vector(v):
    return np.argmax(v)


# run the neural network

training_data, test_data = load_data()
net = NeuralNetwork([784, 30, 10])
net.train(training_data, 30, 10, 3.0, test_data=test_data)

# the network is trained and we can now test it




    

        
