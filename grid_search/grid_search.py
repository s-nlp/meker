import argparse
import wandb

import wandb

from util import import_source_as_module

import_source_as_module('/notebook/Relations_Learning/grid_search/configs/adam_grid.py')

import adam_grid  # python file with default hyperparameters
# Set up your default hyperparameters
hyperparameters = adam_grid

# Pass them wandb.init
wandb.init(config=hyperparameters, project = 'FOxIE')
# Access all hyperparameter values through wandb.config
config = wandb.config
