
from sinusoid import SineMAML
from quadratic import QuadraticMAML
from image_classification import ClassificationMAML

import argparse
import torch
import yaml
import time
import datetime

from utils import MAMLParameters

parser = argparse.ArgumentParser()

parser.add_argument('config', type=str, help='path to configuration file for maml experiment')

args = parser.parse_args()

if __name__ == "__main__":

    with open(args.config, 'r') as yaml_file:
        params = yaml.load(yaml_file, yaml.SafeLoader)

    maml_parameters = MAMLParameters(params) # create object in which to store experiment parameters

    checkpoint_path = 'results/{}/'.format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
    maml_parameters.set_property("checkpoint_path", checkpoint_path)
    
    if torch.cuda.is_available():
        print("Using the GPU")
        maml_parameters.set_property("device", "cuda")
        experiment_device = torch.device("gpu")
    else:
        print("Using the CPU")
        maml_parameters.set_property("device", "cpu")
        experiment_device = torch.device("cpu")

    task = maml_parameters.get("task_type")
    if task == 'sin':
        SM = SineMAML(maml_parameters, experiment_device)
        # SM._generate_batch(plot=True)
        SM.train()
    elif task == 'quadratic':
        QM = QuadraticMAML(maml_parameters)
        QM.train()
    elif task == 'image_classification':
        IM = ClassificationMAML(maml_parameters)
        IM.train()