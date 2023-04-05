import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))



neuralNetwork = NeuralNetwork([2, 3, 1])
print(neuralNetwork.biases)
print(neuralNetwork.weights)
