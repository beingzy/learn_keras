"""implement a simple RNN

   RNN:
   output(t) = activation(W*X(t) + U*state(t) + b)
   state(t) = output(t-1)
"""
import numpy as np


timesteps = 100
input_features = 32
output_features = 64

# generate random input data
inputs = np.random.random((timesteps, input_features))

# initialize state vector
state_t = np.zeros((output_features))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:
    # input_t is a vector of shape (input_features, )
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.concatenate(successive_outputs, axis=0)
