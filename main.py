import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import wandb

from nn import *

# Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess data
val_size = 0.1
train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess_data(train_images, train_labels, test_images, test_labels, val_size)

# Example usage
print("Training Images Shape:", train_images.shape)
print("Validation Images Shape:", val_images.shape)
print("Test Images Shape:", test_images.shape)

NN = NeuralNetwork(28*28, 10, weights_init='xavier', bias_init='zeros')

NN.add_layer(500, activation_type='sigmoid')
NN.add_layer(250, activation_type='sigmoid')
NN.add_layer(100, activation_type='sigmoid')

NN.train(train_images = train_images, train_labels = train_labels, epochs=40, batch_size = 64)