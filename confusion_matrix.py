import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import wandb
from keras.datasets import fashion_mnist
from utils import preprocess_data
from nn import NeuralNetwork
from loss import *
from metrics import calculate_accuracy
from sklearn import metrics

# Load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define validation set size and preprocess
val_size = 0.10
train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess_data(
    train_images, train_labels, test_images, test_labels, val_size
)

NN = NeuralNetwork(28*28, 10, weights_init='xavier_init', bias_init='zeros_init', optimizer="nadam", learning_rate=0.0001, weight_decay=0)

NN.add_layer(128, activation_type='sigmoid')
NN.add_layer(128, activation_type='sigmoid')
NN.add_layer(128, activation_type='sigmoid')

# Load trained model
with open('best_model.pkl', 'rb') as file:
    NN = pickle.load(file)

# Get predictions on test set
pred_label, true_label = NN.test(test_images, test_labels)

# Define class labels
cm_plot_labels = [
    "Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Compute confusion matrix
cm = metrics.confusion_matrix(y_true=true_label, y_pred=pred_label)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

# Convert to DataFrame for visualization
df_cm = pd.DataFrame(cm, index=cm_plot_labels, columns=cm_plot_labels)
df_cm_norm = pd.DataFrame(cm_normalized, index=cm_plot_labels, columns=cm_plot_labels)

# Compute additional model evaluation metrics
precision = metrics.precision_score(true_label, pred_label, average="weighted")
recall = metrics.recall_score(true_label, pred_label, average="weighted")
f1 = metrics.f1_score(true_label, pred_label, average="weighted")

# Create an interactive heatmap using Plotly
fig = go.Figure()

# Add heatmap with actual counts
fig.add_trace(
    go.Heatmap(
        z=df_cm.values,
        x=cm_plot_labels,
        y=cm_plot_labels,
        colorscale="Blues",
        text=df_cm_norm.applymap(lambda x: f"{x:.2%}"),  # Add percentage labels
        texttemplate="%{text}<br>(%{z})",
        hoverinfo="text",
    )
)

test_correct = calculate_accuracy(y_pred = NN.final_output, y = test_labels)
# Update layout for better visualization
fig.update_layout(
    title=f"Confusion Matrix (Test Accuracy = {(test_correct/len(test_labels)):.2%})",
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    xaxis=dict(tickmode="array", tickvals=list(range(len(cm_plot_labels))), ticktext=cm_plot_labels),
    yaxis=dict(tickmode="array", tickvals=list(range(len(cm_plot_labels))), ticktext=cm_plot_labels),
    font=dict(family="Arial, sans-serif", size=12),
    margin=dict(l=100, r=100, t=100, b=100),
)

project_name = 'DA6401 - Assignment1'
entity = 'CE21B031'

run = wandb.init(project = project_name, entity=entity, resume="allow")

# Log results to WandB
wandb.log({
    "Confusion Matrix": fig,
    "Test Accuracy": test_correct/len(test_labels),
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})

# Show plot
fig.show()
