import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="CE21B031",
    # Set the wandb project where this run will be logged.
    project="DA6401 - Assignment1",
)

# Finish the run and upload any remaining data.
run.finish()