import numpy as np

class Loss_CategoricalCrossentropy:
    def __init__(self, output, y):
        self.y_pred = output
        self.y_true = y

    # Calcaulate mean loss
    def calculate(self):
        sample_losses = self.forward()
        data_loss = np.mean(sample_losses)  # Average loss over the batch
        return data_loss
    
    # Calculate the cross entropy loss
    def forward(self):
        samples = len(self.y_pred)
        epsilon = 1e-7
        self.y_pred_clipped = np.clip(self.y_pred, epsilon, 1 - epsilon)  # Avoid log(0)

        if len(self.y_true.shape) == 1:
            # Sparse labels (integer class labels)
            correct_confidences = self.y_pred_clipped[np.arange(samples), self.y_true]
        elif len(self.y_true.shape) == 2:
            # One-hot encoded labels
            correct_confidences = np.sum(self.y_pred_clipped * self.y_true, axis=1)

        # Calculating likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class squared_error():
    """
    Assumes, y_true is one-hot encoded.
    """
    def __init__(self, output, y):
        self.y_pred = output
        self.y_true = y

    def calculate(self):
        loss = np.mean(np.square(self.y_pred - self.y_true))
        return loss