import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import wandb

# Initialize wandb
wandb.init(project='DA6401', entity='ce21b031', name='question1')

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images.astype('float32') / 255.0
train_images = test_images.astype('float32') / 255.0

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Find one sample image for each class
class_indices = [np.where(train_labels == i)[0][0] for i in range(10)]

# Create a 2x5 grid of subplots
rows, cols = 2, 5
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
axes = axes.flatten()

# Plot sample images
for i, idx in enumerate(class_indices):
    axes[i].imshow(train_images[idx], cmap='gray')
    axes[i].set_title(class_names[i])
    axes[i].axis('off')

plt.tight_layout()

# Log the plot to wandb
wandb.log({"Fashion MNIST Samples": wandb.Image(plt)})

plt.show()
