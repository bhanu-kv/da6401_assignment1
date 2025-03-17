from loss import *
import wandb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import wandb
from nn import *

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

# sweep_config = {
#     'method': 'bayes', 
#     'metric': {
#       'name': 'val_acc',
#       'goal': 'maximize'   
#               },
#     'parameters': {
#         'epochs': {
#             'values': [10]
#         },
#         'hidden_layers': {
#             'values': [3,4,5]
#         },
#         'hidden_layer_size' : {
#             'values' : [32,64,128]
#         },
#         'learning_rate': {
#             'values': [0.001,0.0001]
#         },
#         'optimizer': {
#             'values': ["sgd", "mgd", "nag", "rmsprop", "adam","nadam"]
#         },
#         'batch_size': {
#             'values': [64,128]
#         },
#         'weight_init': {
#             'values': ["random_init", "xavier_init"]
#         },
#         'activation_func': {
#             'values': ["sigmoid","tanh","relu"]
#         },
#         'weight_decay': {
#             'values': [0,0.0005,0.5]
#         },
#         'loss_fn': {
#             'values': ["cross_entropy"]
#         }
#     }
# }
sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
              },
    'parameters': {
        'epochs': {
            'values': [10]
        },
        'hidden_layers': {
            'values': [3]
        },
        'hidden_layer_size' : {
            'values' : [128]
        },
        'learning_rate': {
            'values': [0.0001]
        },
        'optimizer': {
            'values': ["sgd", "mgd", "nag", "rmsprop", "adam","nadam"]
        },
        'batch_size': {
            'values': [64]
        },
        'weight_init': {
            'values': ["xavier_init"]
        },
        'activation_func': {
            'values': ["sigmoid"]
        },
        'weight_decay': {
            'values': [0]
        },
        'loss_fn': {
            'values': ["squared_error"]
        }
    }
}

#To add a new agent to an existing sweep, comment next line and directly put sweep_id in wandb.agent
sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

wandb.agent(sweep_id, project=project_name, function=train_wandb)