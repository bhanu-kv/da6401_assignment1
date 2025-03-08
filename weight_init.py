import numpy as np

def weights_initialization(n_inputs, n_neurons, technique = 'random'):
    # Random weight initialization
    if technique == 'random':
        return np.random.rand(n_inputs, n_neurons)
    
    # Xavier Initialization
    if technique == 'xavier':
        N = np.sqrt(6/(n_inputs+n_neurons))

        return np.random.uniform(-N, N, (n_inputs, n_neurons)) 

def bias_initialization(n_neurons, technique):
    # Initializing weights as zero
    if technique == 'zeros':
        return np.zeros((1, n_neurons))