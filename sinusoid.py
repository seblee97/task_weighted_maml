from maml import MAML, ModelNetwork

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class SineMAML(MAML):

    def __init__(self, args, device):
        self.device = device
        self.model = SinusoidalNetwork(args).to(self.device)
        MAML.__init__(self, args)

    def _sample_task(self, amplitude_bounds=(1, 5), phase_bounds=(0, math.pi), domain_bounds=(-5, 5), plot=False):
        amplitude = random.uniform(amplitude_bounds[0], amplitude_bounds[1])
        phase = random.uniform(phase_bounds[0], phase_bounds[1])
        def modified_sin(x):
            return amplitude * np.sin(phase * x)

        if plot:
            fig = plt.figure()
            x = np.linspace(domain_bounds[0], domain_bounds[1], 100)
            y = [modified_sin(xi) for xi in x]
            plt.plot(x, y)
            fig.savefig('sin_batch_test.png')
            plt.close()
        return modified_sin

    def _generate_batch(self, task, domain_bounds=(-5, 5), batch_size=10):
        """
        returns sin function squashed in x direction by a phase parameter sampled randomly between phase_bounds
        enlarged in the y direction by an apmplitude parameter sampled randomly between amplitude_bounds
        """
        x_batch = [random.uniform(domain_bounds[0], domain_bounds[1]) for _ in range(batch_size)]
        y_batch = [task(x) for x in x_batch]
        x_batch_tensor = torch.tensor([[x] for x in x_batch]).to(self.device)
        y_batch_tensor = torch.tensor([[y] for y in y_batch]).to(self.device)
        # print(x_batch_tensor, y_batch_tensor)
        # print(x_batch_tensor.shape, y_batch_tensor.shape)
        return x_batch_tensor, y_batch_tensor

    def _compute_loss(self, prediction, ground_truth):
        loss_function = nn.MSELoss()
        return loss_function(prediction, ground_truth)


class SinusoidalNetwork(ModelNetwork):

    # Make this a more general regression class rather than a Sin specific class?
    def __init__(self, args):
        ModelNetwork.__init__(self, args)

    def construct_layers(self):
        self.linear1 = nn.Linear(self.args.x_dim, 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x