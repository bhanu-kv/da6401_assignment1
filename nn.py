import numpy as np
from weight_init import *
from act_fn import *
from utils import *
from metrics import *
from loss import *
from optimizers import *
from sklearn.metrics import classification_report

class NeuralNetwork:
    def __init__(self, input_size, output_size, weights_init, bias_init, optimizer = 'sgd'):
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

        if optimizer.lower() == 'sgd':
            self.optimizer = SGD(self.learning_rate)
        elif optimizer.lower() == 'adam':
            self.optimizer = Adam(self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
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
        self.optimizer.update(self, X, y)

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

    def generate_classification_report(self, test_dataloader, num_classes=10):
        # Make predictions
        all_preds, all_labels = self.test(test_dataloader)

        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)])
        
        print("Classification Report:\n", report)
            