
from sinusoid import SineMAML
from quadratic import QuadraticMAML
from image_classification import ClassificationMAML

import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('-task_type', default=None, type=str, help='which task to meta-learn e.g. -sin- for sinusoid regression')
parser.add_argument('-training_iterations', default=250000, help='number of training iterations (total calls to the outer training loop)')
parser.add_argument('-task_batch_size', default=25, help='number of tasks sampled per meta-update (per outer loop)')
parser.add_argument('-meta_lr', default=0.001, help='the base learning rate of the generator (the outer loop optimiser)')
parser.add_argument('-inner_update_batch_size', default=10, help='number of examples used for inner gradient update (K for K-shot learning)')
parser.add_argument('-inner_update_lr', default=0.001, help='step size alpha for inner gradient update')
parser.add_argument('-num_inner_updates', default=10, help='number of inner gradient updates during training')
parser.add_argument('-x_dim', default=1, help='dimension of x') #TODO
parser.add_argument('-validation_task_batch_size', default=10, help='number of tasks to sample and evaluate in each validation loop')
parser.add_argument('-validation_frequency', default=200, help='frequency with which to perform validation during training')

args = parser.parse_args()

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        print("Using the GPU")
        experiment_device = torch.device("cpu")
    else:
        print("Using the CPU")
        experiment_device = torch.device("cpu")

    if args.task_type == 'sin':
        SM = SineMAML(args, experiment_device)
        # SM._generate_batch(plot=True)
        SM.train()
    elif args.task_type == 'quadratic':
        QM = QuadraticMAML(args)
        QM.train()
    elif args.task_type == 'image_classification':
        IM = ClassificationMAML(args)
        IM.train()