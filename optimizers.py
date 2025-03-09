import numpy as np
from act_fn import *

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, network):
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent
    """

    def update(self, network, X, y):
        # Error in ifnal result
        error = network.final_output - y

        # Compute gradients and updating weights
        for i in range(network.hidden_size, -1, -1):
            # When final layer error is final result error
            if (i==network.hidden_size):
                layer_error = error
            # Else error is backpropogated
            else:
                layer_error = np.dot(layer_error, network.weights[i+1].T) * der_activation_func(network.outputs_with_act[i], type = network.activation[i])

            if i > 0:
                inputs = network.outputs_with_act[i - 1]
            else:
                inputs = X

            # Calculating Gradient
            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            # Update weights and biases
            network.weights[i] -= grad_w*self.learning_rate
            network.bias[i] -= grad_b*self.learning_rate

class MGD(Optimizer):
    """
    Momentum-based Gradient Descent
    """
    def __init__(self, momentum=0.9):
        '''
        momentum: Beta term used for weighing Velocity at t-1
        '''
        # Initializing Components of MGD
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, network, X, y):
        '''
        Function used to update weights and biases
        '''
        # Initializing Velocity terms with same dimensions as w and b
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in network.weights]
            self.velocity_b = [np.zeros_like(b) for b in network.bias]

        # Error in ifnal result
        error = network.final_output - y

        # Compute gradients and updating weights
        for i in range(network.hidden_size, -1, -1):
            # When final layer error is final result error
            if (i==network.hidden_size):
                layer_error = error
            # Else error is backpropogated
            else:
                layer_error = np.dot(layer_error, network.weights[i+1].T) * der_activation_func(network.outputs_with_act[i], type = network.activation[i])

            if i > 0:
                inputs = network.outputs_with_act[i - 1]
            else:
                inputs = X

            # Calculating Gradient
            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            # Update velocity
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * grad_w
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * grad_b

            # Update weights and biases
            network.weights[i] += self.velocity_w[i]
            network.bias[i] += self.velocity_b[i]

class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG)
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, network, X, y):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in network.weights]
            self.velocity_b = [np.zeros_like(b) for b in network.bias]

        # Error in ifnal result
        error = network.final_output - y

        # Compute gradients and updating weights
        for i in range(network.hidden_size, -1, -1):
            # When final layer error is final result error
            if (i==network.hidden_size):
                layer_error = error
            # Else error is backpropogated
            else:
                layer_error = np.dot(layer_error, network.weights[i+1].T) * der_activation_func(network.outputs_with_act[i], type = network.activation[i])

            if i > 0:
                inputs = network.outputs_with_act[i - 1]
            else:
                inputs = X

            # Calculating Gradient
            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            # Update velocity with Nesterov correction
            prev_velocity_w = np.copy(self.velocity_w[i])
            prev_velocity_b = np.copy(self.velocity_b[i])

            # Using lookahead term for updating velocities
            self.velocity_w[i] = self.momentum * prev_velocity_w + self.learning_rate * (grad_w - self.momentum * prev_velocity_w)
            self.velocity_b[i] = self.momentum * prev_velocity_b + self.learning_rate * (grad_b - self.momentum * prev_velocity_b)

            # Update weights and biases
            network.weights[i] -= self.velocity_w[i]
            network.bias[i] -= self.velocity_b[i]