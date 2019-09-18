
from context import maml, utils, jax_maml

import argparse
import torch
import yaml
import time
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('-base_config', type=str, help='path to base configuration file for maml experiment', default='configs/base_config.yaml')
parser.add_argument('-config', type=str, help='path to specific configuration file for maml experiment')
parser.add_argument('-framework', type=str, help='jax or pytorch model', default='jax')

args = parser.parse_args()

if __name__ == "__main__":

    # base parameters common to all configs
    with open(args.base_config, 'r') as base_yaml_file:
        base_params = yaml.load(base_yaml_file, yaml.SafeLoader)

    # specific parameters
    with open(args.config, 'r') as yaml_file:
        specific_params = yaml.load(yaml_file, yaml.SafeLoader)

    maml_parameters = utils.parameters.MAMLParameters(base_params) # create object in which to store experiment parameters

    # update base maml parameters with specific parameters
    maml_parameters.update(specific_params)

    exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = maml_parameters.get("experiment_name")
    if experiment_name:
        checkpoint_path = 'results/{}/{}/'.format(exp_timestamp, experiment_name)
    else:
        checkpoint_path = 'results/{}/'.format(exp_timestamp)
    maml_parameters.set_property("checkpoint_path", checkpoint_path)
    maml_parameters.set_property("experiment_timestamp", exp_timestamp)
    maml_parameters.set_property("framework", args.framework)

    seed_value = maml_parameters.get("seed")
    
    # TODO: set seeds correctly, does it need to be done separately for each script? Look at script dependencies
    import random
    import numpy as np
    import torch

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available() and maml_parameters.get('use_gpu'):
        print("Using the GPU")
        maml_parameters.set_property("device", "cuda")
        experiment_device = torch.device("gpu")
    else:
        print("Using the CPU")
        maml_parameters.set_property("device", "cpu")
        experiment_device = torch.device("cpu")

    task = maml_parameters.get("task_type")
    if 'sin' in task:
        if args.framework == 'pytorch':
            SM = maml.sinusoid.SineMAML(maml_parameters, experiment_device)
        elif args.framework == 'jax':
            SM = jax_maml.jax_sinusoid.SineMAML(maml_parameters, experiment_device)
        else:
            raise ValueError("Invalid framework argument. Use 'jax' or 'pytorch'")
        SM.train()
    elif task == 'quadratic':
        QM = maml.quadratic.QuadraticMAML(maml_parameters)
        QM.train()
    elif task == 'image_classification':
        IM = maml.image_classification.ClassificationMAML(maml_parameters)
        IM.train()