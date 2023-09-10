import pandas as pd
import numpy as np
from sympy import diff, symbols, exp

df = pd.read_csv('../data/clean_weather.csv')

feature_array = df["tmax"].head(15).to_numpy()[np.newaxis, :]

np.random.seed(0)


# Initialize the weights and biases
input_weights = np.random.rand(1, 5)/5 - 0.1

hidden_weights = np.random.rand(5, 5)/5 - 0.1
hidden_bias = np.random.rand(1, 5)/5 - 0.1

output_weights = np.random.rand(5, 1) * 50
output_bias = np.random.rand(1, 1)

input_values = feature_array[0, 7:10]

outputs = np.zeros(3)
hiddens = np.zeros((3, 5))
prev_hidden = None
input_sequence = feature_array[0, 7:10]
print(input_sequence)

#FORWARDPASS
for i in range(3):
    x = input_sequence[i].reshape(1, 1)
    xi = np.dot(x, input_weights)

    if prev_hidden is not None:
        xh =  np.dot(prev_hidden, hidden_weights) + xi + hidden_bias
    else:
        xh = xi

    
    xh = np.tanh(xh)
    prev_hidden = xh
    hiddens[i,] = xh
    xo = np.dot(xh, output_weights) + output_bias
    outputs[i] = xo

#BACKWARD PASS 
#using mean square error


