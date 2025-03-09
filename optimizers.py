import numpy as np
from act_fn import *

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, network):
        raise NotImplementedError

class SGD(Optimizer):
    def update(self, network, X, y):
        error = network.final_output - y

        # Compute the hidden layer error
        for i in range(network.hidden_size, -1, -1):
            if (i==network.hidden_size):
                layer_error = error
            else:
                layer_error = np.dot(layer_error, network.weights[i+1].T) * der_activation_func(network.outputs_with_act[i], type = network.activation[i])

            if i > 0:
                inputs = network.outputs_with_act[i - 1]
            else:
                inputs = X

            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            network.weights[i] -= grad_w*network.learning_rate
            network.bias[i] -= grad_b*network.learning_rate