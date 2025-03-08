import numpy as np
from weight_init import *
from act_fn import *
from utils import *
from metrics import *
from loss import *

# import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

class NeuralNetwork:
    def __init__(self, input_size, output_size, weights_init, bias_init):
        # Initializing number of inputs and outputs and the number of layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 0

        # Initializing Weights and Bias Initializing techniques
        self.weights_init = weights_init
        self.bias_init = bias_init
        
        # Assigning Initial Weights, Bias and Activation Function for all the layers
        self.weights = [weights_initialization(self.input_size, self.output_size, technique=weights_init)]
        self.bias = [bias_initialization(self.output_size, technique=bias_init)]
        self.activation = ['linear']
        
        # Initializing all the required variables as None
        self.input = None
        self.final_output = None

        self.learning_rate = 0.01
        self.outputs_with_act = []
        self.outputs_without_act = []
        
    def add_layer(self, no_neurons, activation_type= 'ReLU'):
        # Number of hidden layers increase by one after addition of every layer
        self.hidden_size += 1
        prev_layer_shape = self.weights[-1].shape

        # Initialization of new weights and shift of previous weights to output layer
        self.weights[-1] = weights_initialization(prev_layer_shape[0], no_neurons, self.weights_init)
        self.weights.append(weights_initialization(no_neurons, prev_layer_shape[1], self.weights_init))

        # Initialization of new bias and shift of previous bias to output layer
        self.bias[-1] = bias_initialization(no_neurons, self.bias_init)
        self.bias.append(bias_initialization(prev_layer_shape[1], self.bias_init))

        # Adding new activation function
        prev_activation = self.activation[-1]

        self.activation[-1] = activation_type
        self.activation.append(prev_activation)
    
    def forward(self, inputs):
        self.inputs = inputs
        layer_output = inputs

        # Storing Output before and after applying activation function
        self.outputs_with_act = []
        self.outputs_without_act = []

        # Forward Propogation in a loop for each layer
        for i in range(self.hidden_size+1):
            # Calculating outputs without activation function
            layer_output = np.dot(layer_output, self.weights[i]) + self.bias[i]
            self.outputs_without_act.append(layer_output)

            # Calculating Outputs with activation
            layer_output = activation_func(layer_output, type = self.activation[i])
            self.outputs_with_act.append(layer_output)
        
        # Applying softmax for final layer to calculate probabilities
        softmax = Activation_Softmax(layer_output)
        self.final_output = softmax.forward()

    def backprop(self, X, y):
        error = self.final_output - y
        partial_derivative_w = np.dot(self.outputs_with_act[-2].T, error) / y.shape[0]
        partial_derivative_b = np.sum(error, axis=0, keepdims=True) / y.shape[0]

        self.weights[-1] -= partial_derivative_w*self.learning_rate
        self.bias[-1] -= partial_derivative_b*self.learning_rate

        # Compute the hidden layer error
        for i in range(self.hidden_size-1, -1, -1):
            error = np.dot(error, self.weights[i+1].T) * der_activation_func(self.outputs_with_act[i], type = self.activation[i])

            if i>0:
                partial_derivative_w = np.dot(self.outputs_with_act[i-1].T, error) / y.shape[0]
                partial_derivative_b = np.sum(error, axis=0, keepdims=True) / y.shape[0]
            else:
                partial_derivative_w = np.dot(X.T, error) / y.shape[0]
                partial_derivative_b = np.sum(error, axis=0, keepdims=True) / y.shape[0]

            self.weights[i] -= partial_derivative_w*self.learning_rate
            self.bias[i] -= partial_derivative_b*self.learning_rate


    def train(self, train_images, train_labels, epochs, batch_size=1):     
        for epoch in range(epochs):
            train_loss = 0
            train_correct = 0
            
            for i in range(0, len(train_images), batch_size):
                batch_images = train_images[i:min(train_images.shape[0],i+batch_size)]
                batch_labels = train_labels[i:min(train_images.shape[0],i+batch_size)]

                self.forward(batch_images)
                loss_fn = Loss_CategoricalCrossentropy(output= self.final_output, y = batch_labels)
                loss = loss_fn.calculate()
                correct = calculate_accuracy(y_pred = self.final_output, y = batch_labels)

                train_loss += loss
                train_correct += correct
                
                self.backprop(batch_images, batch_labels)
                
            if (epoch+1) % 1 == 0:
                print("------------------------------------------------------")
                print("Epoch:", epoch+1)
                print("Loss:", train_loss)
                print("Correct Predications:", train_correct)
                print("Total Images:", len(train_images))
                print("Accuracy:", train_correct/len(train_images))
                print()

            # for (images, labels) in tqdm(train_dataloader, desc="Training", leave=False):
            #     images = images.view(images.size(0), -1).numpy()  # Flatten each image

            #     num_classes = 10
            #     labels = torch.nn.functional.one_hot(labels, num_classes)
            #     labels = labels.cpu().numpy()
            #     # labels = np.eye(self.output_size)[labels.numpy()]

            #     images = normalize_images(images)   
            #     self.forward(images)
                
            #     loss_fn = Loss_CategoricalCrossentropy(output= self.final_output, y = labels)
            #     loss = loss_fn.calculate()
            #     correct = calculate_accuracy(y_pred = self.final_output, y = labels)

            #     train_loss += loss
            #     train_correct += correct
                
            #     self.backprop(images, labels)
                
            # if (epoch+1) % 1 == 0:
            #     print("------------------------------------------------------")
            #     print("Epoch:", epoch+1)
            #     print("Loss:", train_loss)
            #     print("Correct Predications:", train_correct)
            #     print("Total Images:", len(train_dataloader.dataset))
            #     print("Accuracy:", train_correct/len(train_dataloader.dataset))
            #     print()
    
    # def test(self, test_dataloader):
    #     all_preds = []
    #     all_labels = []

    #     for (images, labels) in tqdm(test_dataloader, desc="Testing", leave=False):
    #         images = images.view(images.size(0), -1).numpy()  # Flatten each image

    #         num_classes = 10
    #         labels = torch.nn.functional.one_hot(labels, num_classes)
    #         labels = labels.cpu().numpy()

    #         images = normalize_images(images)
    #         self.forward(images)

    #         top_pred = self.final_output.argmax(1, keepdims=True)
    #         y_correct = labels.argmax(1, keepdims=True)

    #         all_preds.append(top_pred)
    #         all_labels.append(y_correct)
        
    #     # Convert list of arrays into single arrays
    #     all_preds = np.concatenate(all_preds, axis=0)
    #     all_labels = np.concatenate(all_labels, axis=0)

    #     return all_preds, all_labels

    def generate_classification_report(self, test_dataloader, num_classes=10):
        # Make predictions
        all_preds, all_labels = self.test(test_dataloader)

        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)])
        
        print("Classification Report:\n", report)
            