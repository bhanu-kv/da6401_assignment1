import numpy as np

def calculate_accuracy(y_pred, y):
    # Calculate accuracy by matching the outputs of predicted and true values
    top_pred = y_pred.argmax(1, keepdims=True)
    y_correct = y.argmax(1, keepdims=True)
    
    correct = np.sum(top_pred == y_correct)

    return correct