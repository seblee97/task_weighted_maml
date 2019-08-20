
from context import maml, utils

import argparse
import torch
import yaml
import time
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, help='path to configuration file for maml experiment')

args = parser.parse_args()

if __name__ == "__main__":

    # base parameters common to all configs
    with open('configs/base_config.yaml', 'r') as base_yaml_file:
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
    if task == 'sin':
        SM = maml.sinusoid.SineMAML(maml_parameters, experiment_device)
        # SM._generate_batch(plot=True)
        SM.train()
    elif task == 'quadratic':
        QM = maml.quadratic.QuadraticMAML(maml_parameters)
        QM.train()
    elif task == 'image_classification':
        IM = maml.image_classification.ClassificationMAML(maml_parameters)
        IM.train()