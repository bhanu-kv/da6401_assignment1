# DA6401 Assignment 1

[Project Report on Weights & Biases](https://wandb.ai/ce21b031/DA6401%20-%20Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTgzMjM5Mg?accessToken=o241ydj5v3miydwoidt9zxcdhl8xgx4d9aig6ecoahu0zq04uzq4kbo871yqqjfb)

## Overview
This project implements a Neural Network from scratch using only NumPy and other Python data structures. The model is tested on multiclass classification datasets such as Fashion MNIST and MNIST.

## Code Organization

act_fn.py: Contains all the activation functions
confusion_matrix.py: Used to plot confusion matrix
loss.py: Contains all the loss functions
metrics.py: Contains all the metrics
optimizers.py: Contains all Optimizer Classes
q1_plot_images.py: Used to plot images of each class
train_fashion.py: Trained fashion mnist dataset
train_mnist.py: Trained mnist dataset
train.py: Used for python sweep with arguement parser
nn.py: Contains Neural Network Class
utils: Contains all the util functions
weight_init.py: Contains Weight Initialization functions
wandb_sweep.py: Sweep without arguement parser

## Defining the Neural Network
To initialize the neural network, use the following:

```python
NN = NeuralNetwork(
    input_size=28*28,
    output_size=10,
    weights_init="xavier_init",
    bias_init="random_init",
    optimizer='adam',
    learning_rate=0.001,
    momentum=0.9,
    beta=0.9,
    epsilon=1e-8,
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.0005,
    loss_fun="cross_entropy"
)
```

## Adding a Hidden Layer
To add a hidden layer with 128 neurons and a sigmoid activation function:

```python
NN.add_layer(128, activation_type='sigmoid')
```

## Implementing a Custom Optimizer
To define a new optimizer, modify the `update` function in `optimizer.py` to include Weight and Bias Update Formulas:

```python
class NewOptimizer(Optimizer):
    """
    Custom Optimizer
    """
    def __init__(self, network):
        self.learning_rate = network.learning_rate
        self.weight_decay = network.weight_decay

    def update(self, network, X, y):
        error = network.final_output - y

        for i in range(network.hidden_size, -1, -1):
            if i == network.hidden_size:
                layer_error = error
            else:
                layer_error = np.dot(layer_error, network.weights[i+1].T) * der_activation_func(
                    network.outputs_with_act[i], type=network.activation[i]
                )

            inputs = network.outputs_with_act[i - 1] if i > 0 else X
            grad_w = np.dot(inputs.T, layer_error) / y.shape[0]
            grad_b = np.sum(layer_error, axis=0, keepdims=True) / y.shape[0]

            network.weights[i] += # Weight update formula
            network.bias[i] += # Bias update formula
```

## Training the Neural Network
To train the model:
1. Define the network architecture.
2. Load the dataset.
3. Train using:

```python
NN.train(
    train_images=train_images,
    train_labels=train_labels,
    val=(val_images, val_labels),
    epochs=no_epochs,
    batch_size=batch_size
)
```

## Running a Weights & Biases (WandB) Sweep
To perform a hyperparameter sweep, run:

```sh
python train.py --wandb_entity myname --wandb_project myprojectname
```

### Sweep Argument Options
| Argument | Default Value | Description |
|----------|--------------|-------------|
| `-wp, --wandb_project` | `myprojectname` | Project name in Weights & Biases dashboard |
| `-we, --wandb_entity` | `myname` | WandB entity name |
| `-d, --dataset` | `fashion_mnist` | Dataset (`mnist`, `fashion_mnist`) |
| `-e, --epochs` | `1` | Number of training epochs |
| `-b, --batch_size` | `4` | Batch size |
| `-l, --loss` | `cross_entropy` | Loss function (`mean_squared_error`, `cross_entropy`) |
| `-o, --optimizer` | `sgd` | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) |
| `-lr, --learning_rate` | `0.1` | Learning rate |
| `-m, --momentum` | `0.5` | Momentum for `momentum` and `nag` optimizers |
| `-beta, --beta` | `0.5` | Beta for `rmsprop` optimizer |
| `-beta1, --beta1` | `0.5` | Beta1 for `adam` and `nadam` optimizers |
| `-beta2, --beta2` | `0.5` | Beta2 for `adam` and `nadam` optimizers |
| `-eps, --epsilon` | `0.000001` | Epsilon for optimizers |
| `-w_d, --weight_decay` | `0.0` | Weight decay |
| `-w_i, --weight_init` | `random` | Weight initialization (`random`, `Xavier`) |
| `-nhl, --num_layers` | `1` | Number of hidden layers |
| `-sz, --hidden_size` | `4` | Number of hidden neurons per layer |
| `-a, --activation` | `sigmoid` | Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`) |

Use these options to configure and run a hyperparameter sweep efficiently.