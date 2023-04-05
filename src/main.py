import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the data
data_train = pd.read_csv('/Users/harmyabhatt/NeuralNetworkFromScratch/data/mnist_train.csv')
data_test = pd.read_csv('/Users/harmyabhatt/NeuralNetworkFromScratch/data/mnist_test.csv')

# Convert the data into numpy arrays
train_data = data_train.values
test_data = data_test.values

#



