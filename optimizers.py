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
    def __init__(self, learning_rate = 0.01, momentum=0.9):
        # Initializing Components of MGD
        super().__init__(learning_rate)

    def update(self, network, X, y):
        # Error in final result
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
    def __init__(self, learning_rate = 0.01, momentum=0.9):
        '''
        momentum: Beta term used for weighing Velocity at t-1
        '''
        # Initializing Components of MGD
        super().__init__(learning_rate)
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

        # Error in final result
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
        '''
        momentum: Beta term used for weighing Velocity at t-1
        '''
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, network, X, y):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in network.weights]
            self.velocity_b = [np.zeros_like(b) for b in network.bias]

        # Error in final result
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

class RMSprop(Optimizer):
    """
    RMSprop Optimizer
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        # Initializing Components of MGD
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.velocity_w = None
        self.velocity_b = None

    def update(self, network, X, y):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in network.weights]
            self.velocity_b = [np.zeros_like(b) for b in network.bias]

        # Error in final result
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

            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            # Update squared gradients with RMSprop rule
            self.velocity_w[i] = self.beta * self.velocity_w[i] + (1 - self.beta) * (grad_w ** 2)
            self.velocity_b[i] = self.beta * self.velocity_b[i] + (1 - self.beta) * (grad_b ** 2)

            # Update weights and biases
            network.weights[i] -= (self.learning_rate / (np.sqrt(self.velocity_w[i]) + self.epsilon)) * grad_w
            network.bias[i] -= (self.learning_rate / (np.sqrt(self.velocity_b[i]) + self.epsilon)) * grad_b

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, network, X, y):
        if self.m is None:
            self.m_w = [np.zeros_like(w) for w in network.weights]
            self.v_w = [np.zeros_like(w) for w in network.weights]

            self.m_b = [np.zeros_like(b) for b in network.bias]
            self.v_b = [np.zeros_like(b) for b in network.bias]

        self.t += 1
        
        # Error in final result
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

            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            # Update biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b**2)
            m_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat = self.v_b[i] / (1 - self.beta2**self.t)
            network.bias[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Update weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w**2)
            m_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat = self.v_w[i] / (1 - self.beta2**self.t)
            network.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Nadam(Optimizer):
    """
    Nadam Optimizer (Nesterov-accelerated Adaptive Moment Estimation)
    Algorithm Source: https://optimization.cbe.cornell.edu/index.php?title=Nadam
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Time step

    def update(self, network, X, y):
        if self.m is None:
            # Initialize first and second moment vectors for weights and biases
            self.m_w = [np.zeros_like(w) for w in network.weights]
            self.v_w = [np.zeros_like(w) for w in network.weights]

            self.m_b = [np.zeros_like(b) for b in network.bias]
            self.v_b = [np.zeros_like(b) for b in network.bias]

        self.t += 1  # Increment time step
        
        # Error in final result
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

            # Compute gradients
            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            # Update biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b**2)

            m_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat = self.v_b[i] / (1 - self.beta2**self.t)

            network.bias[i] -= self.learning_rate * (self.beta1 * m_hat + (1 - self.beta1) * grad_b) / (np.sqrt(v_hat) + self.epsilon)

            # Update weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w**2)

            m_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat = self.v_w[i] / (1 - self.beta2**self.t)

            network.weights[i] -= self.learning_rate * (self.beta1 * m_hat + (1 - self.beta1) * grad_w) / (np.sqrt(v_hat) + self.epsilon)
