import numpy as np

# Activation Functions
def activation_func(inputs, type = 'sigmoid'):
    if type == 'ReLU':
        # return np.maximum(0, inputs)
        return np.minimum(np.maximum(0, inputs), 100)
    elif type == 'sigmoid':
        return 1 / (1 + np.exp(-inputs))
    elif type == 'linear':
        return inputs
    elif type == 'tanh':
        return (np.exp(inputs)-np.exp(-inputs))/(np.exp(inputs)+np.exp(-inputs))

# Derivative of Activation Functions
def der_activation_func(inputs, type = 'sigmoid'):
    if type == 'ReLU':
        output = np.where(inputs >= 0, 0.0, 1.0)
        # output = output.astype(float)
        
        return output

    elif type == 'sigmoid':
        # inputs = activation_func(inputs, type = 'sigmoid')
        return inputs*(1-inputs)
    
    elif type == 'tanh':
        return 1 - np.square(activation_func(inputs, type = 'tanh'))

    elif type == 'linear':
        return inputs
    
# Softmax to calculate probabilities
class Activation_Softmax:
    def __init__(self, inputs):
        self.inputs = inputs

    def forward(self):
        exp_values = np.exp(self.inputs - np.max(self.inputs, axis=1, keepdims=True)) # # Subratcting max to ensure numerical stability (overflow of large exp)
        exp_sum = np.sum(exp_values, axis=1, keepdims=True)
        probabilities = exp_values / exp_sum

        return probabilities