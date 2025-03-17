import numpy as np

def weights_initialization(n_inputs, n_neurons, technique = 'random_init'):
    # Random weight initialization
    if technique == 'random_init':
        return np.random.rand(n_inputs, n_neurons)
    
    # Xavier Initialization
    if technique == 'xavier_init':
        N = np.sqrt(6/(n_inputs+n_neurons))

        return np.random.uniform(-N, N, (n_inputs, n_neurons)) 

def bias_initialization(n_neurons, technique):
    # Initializing weights as zero
    if technique == 'zeros_init':
        return np.zeros((1, n_neurons))