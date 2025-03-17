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

NN = NeuralNetwork(28*28, 10, weights_init='xavier_init', bias_init='zeros_init', optimizer="nadam", learning_rate=0.0001, weight_decay=0)

NN.add_layer(128, activation_type='sigmoid')
NN.add_layer(128, activation_type='sigmoid')
NN.add_layer(128, activation_type='sigmoid')

# with open('best_model.pkl', 'rb') as file:
#     NN = pickle.load(file)

NN.train(train_images = train_images, train_labels = train_labels, val = (val_images, val_labels), epochs=40, batch_size = 64)
# NN.train(train_images = train_images, train_labels = train_labels, val = None, epochs=40, batch_size = 64)
# NN.test(test_images = test_images, test_labels = test_labels)
NN.generate_classification_report(test_images, test_labels)