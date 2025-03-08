import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

def normalize_images(images):
    """Normalizes pixel values to the range [0, 1]."""
    return images.astype('float32') / 255.0

def flatten_image_array(images):
    """Flattens 28x28 images into 784-dimensional vectors."""
    return images.reshape(images.shape[0], -1)

def encode_labels(labels, num_classes):
    """One-hot encodes labels."""
    return np.eye(num_classes)[labels]

def split_validation_set(images, labels, val_size):
    """Splits data into training and validation sets."""
    return train_test_split(images, labels, test_size=val_size, random_state=1)

def preprocess_data(train_images, train_labels, test_images, test_labels, val_size):
    """
    Preprocesses data by normalizing images, flattening them, 
    and one-hot encoding labels. Also creates a validation set.
    """
    num_classes = len(np.unique(train_labels))
    
    # Normalize images
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)
    
    # Flatten images
    train_images = flatten_image_array(train_images)
    test_images = flatten_image_array(test_images)
    
    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = split_validation_set(train_images, train_labels, val_size)
    
    # One-hot encode labels
    train_labels = encode_labels(train_labels, num_classes)
    val_labels = encode_labels(val_labels, num_classes)
    test_labels = encode_labels(test_labels, num_classes)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
