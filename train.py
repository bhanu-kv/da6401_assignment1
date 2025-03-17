from loss import *
import wandb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import wandb
from nn import *
import argparse

def train_wandb(config = None):
    # Load Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Preprocess data
    val_size = 0.1
    train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess_data(train_images, train_labels, test_images, test_labels, val_size)

    run = wandb.init(config=config, resume="allow")
    config = wandb.config

    name = f'hl_{config.hidden_layers}_bs_{config.batch_size}_acf_{config.activation_func}_lr_{config.learning_rate}_opt_{config.optimizer}_w_init_{config.weight_init}_wdecay_{config.weight_decay}'
    wandb.run.name = name
    wandb.run.save()

    NN = NeuralNetwork(input_size = 28*28,
                       output_size = 10,
                       weights_init=config.weight_init,
                       bias_init='zeros_init',
                       optimizer = config.optimizer,
                       learning_rate=config.learning_rate,
                       weight_decay = config.weight_decay,
                       loss_fun = config.loss_fn
                       )

    for i in range(config.hidden_layers) :
        NN.add_layer(config.hidden_layer_size, activation_type=config.activation_func)

    
    NN.train(train_images = train_images,
             train_labels = train_labels,
             val = (val_images, val_labels),
             epochs=config.epochs,
             batch_size = config.batch_size
            )
    
    wandb.finish()


project_name = 'DA6401 - Assignment1'
entity = 'CE21B031'

def main():
    parser = argparse.ArgumentParser(description="Train a Neural Network with WandB Sweeps")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", help="Project name for WandB")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="WandB entity")
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")
    
    args = parser.parse_args()
    
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': vars(args)
    }
    
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    wandb.agent(sweep_id, function=train_wandb)

if __name__ == "__main__":
    main()