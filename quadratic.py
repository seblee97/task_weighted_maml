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

class QuadraticMAML(MAML):

    def __init__(self, args):
        self.model = QuadraticNetwork(args)
        MAML.__init__(self, args)

    def _sample_task(self, quadratic_bounds=(-2, 2), linear_bounds=(-2, 2), constant_bounds=(-2, 2), domain_bounds=(-5, 5), plot=False):
        quadratic_term = random.uniform(quadratic_bounds[0], quadratic_bounds[1])
        linear_term = random.uniform(linear_bounds[0], linear_bounds[1])
        constant_term = random.uniform(constant_bounds[0], constant_bounds[1])
        def quadratic_function(x):
            return quadratic_term * x ** 2 + linear_term * x + constant_term

        if plot:
            fig = plt.figure()
            x = np.linspace(domain_bounds[0], domain_bounds[1], 100)
            y = [quadratic_function(xi) for xi in x]
            plt.plot(x, y)
            fig.savefig('quadratic_batch_test.png')
            plt.close()
        return quadratic_function

    def _generate_batch(self, task, domain_bounds=(-5, 5), batch_size=10):
        """
        returns sin function squashed in x direction by a phase parameter sampled randomly between phase_bounds
        enlarged in the y direction by an apmplitude parameter sampled randomly between amplitude_bounds
        """
        x_batch = [random.uniform(domain_bounds[0], domain_bounds[1]) for _ in range(batch_size)]
        y_batch = [task(x) for x in x_batch]
        x_batch_tensor = torch.tensor([[x] for x in x_batch])
        y_batch_tensor = torch.tensor([[y] for y in y_batch])
        # print(x_batch_tensor, y_batch_tensor)
        # print(x_batch_tensor.shape, y_batch_tensor.shape)
        return x_batch_tensor, y_batch_tensor

    def _compute_loss(self, prediction, ground_truth):
        loss_function = nn.MSELoss()
        return loss_function(prediction, ground_truth)

class QuadraticNetwork(ModelNetwork):

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