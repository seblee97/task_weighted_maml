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

from typing import Any, Dict, List

class SineMAML(MAML):

    def __init__(self, params, device):
        self.device = device

        # extract relevant task-specific parameters
        self.amplitude_bounds = params.get(['sin', 'amplitude_bounds'])
        self.domain_bounds = params.get(['sin', 'domain_bounds'])
        degree_phase_bounds = params.get(['sin', 'phase_bounds']) # phase given in degrees
        block_sizes = params.get(['sin', 'fixed_val_blocks']) 

        # convert phase bounds/ fixed_val_interval from degrees to radians
        self.phase_bounds = [
            degree_phase_bounds[0] * (2 * np.pi) / 360, degree_phase_bounds[1] * (2 * np.pi) / 360
            ]
        
        self.block_sizes = [block_sizes[0], block_sizes[1] * (2 * np.pi) / 360]

        self.model_inner = SinusoidalNetwork(params).to(self.device)

        MAML.__init__(self, params)

    def _sample_task(self, plot=False):
        """
        returns sin function squashed in x direction by a phase parameter sampled randomly between phase_bounds
        enlarged in the y direction by an apmplitude parameter sampled randomly between amplitude_bounds
        """
        amplitude = random.uniform(self.amplitude_bounds[0], self.amplitude_bounds[1])
        phase = random.uniform(self.phase_bounds[0], self.phase_bounds[1])
        def modified_sin(x):
            return amplitude * np.sin(phase * x)
        return modified_sin

    def _get_task_from_params(self, parameters: List[float]) -> Any:
        """
        Return sine function defined by parameters given

        :param parameters: parameters defining the specific sin task in the distribution

        :return modified_sin: sin function

        (method differs from _sample_task in that it is not a random sample but
        defined by parameters given)
        """
        def modified_sin(x):
            return parameters[0] * np.sin(parameters[1] * x)
        return modified_sin

    def visualise(self, model_iterations, task, validation_x, validation_y, save_name):

        dummy_model = SinusoidalNetwork(self.params)

        # ground truth
        plot_x = np.linspace(self.domain_bounds[0], self.domain_bounds[1], 100)
        plot_x_tensor = torch.tensor([[x] for x in plot_x]).to(self.device)
        plot_y_ground_truth = [task(xi) for xi in plot_x]

        fig = plt.figure()
        plt.plot(plot_x, plot_y_ground_truth, label="Ground Truth")

        for i, (model_weights, model_biases) in enumerate(model_iterations):
            
            dummy_model.weights = model_weights
            dummy_model.biases = model_biases

            plot_y_prediction = dummy_model(plot_x_tensor)
            plt.plot(plot_x, plot_y_prediction.cpu().detach().numpy(), linestyle='dashed', label='Fine-tuned MAML {} update'.format(i))

        plt.scatter(validation_x.cpu(), validation_y.cpu(), marker='o', label='K Points')

        plt.title("Validation of Sinusoid Meta-Regression")
        plt.xlabel(r"x")
        plt.ylabel(r"sin(x)")
        plt.legend()
        
        # fig.savefig(self.params.get("checkpoint_path") + save_name)
        plt.close()

        return fig

    def _generate_batch(self, task, batch_size=10, plot=False): # Change batch generation to be done in pure PyTorch
        """
        generates an array, x_batch, of B datapoints sampled randomly between domain_bounds
        and computes the sin of each point in x_batch to produce y_batch.
        """
        x_batch = [random.uniform(self.domain_bounds[0], self.domain_bounds[1]) for _ in range(batch_size)]
        y_batch = [task(x) for x in x_batch]
        x_batch_tensor = torch.tensor([[x] for x in x_batch]).to(self.device)
        y_batch_tensor = torch.tensor([[y] for y in y_batch]).to(self.device)
        # print(x_batch_tensor, y_batch_tensor)
        # print(x_batch_tensor.shape, y_batch_tensor.shape)
        # print(x_batch_tensor.dtype)       

        if plot:
            fig = plt.figure()
            x = np.linspace(self.domain_bounds[0], self.domain_bounds[1], 100)
            y = [task(xi) for xi in x]
            plt.plot(x, y)
            fig.savefig('sin_batch_test.png')
            plt.close()
        return x_batch_tensor, y_batch_tensor

    def _get_fixed_validation_tasks(self):
        """
        If using fixed validation this method returns a set of tasks that are 
        'representative' of the task distribution in some meaningful way. 

        In the case of sinusoidal regression we split the parameter space equally.
        """
        # mesh of equally partitioned state space
        amplitude_spectrum, phase_spectrum = np.mgrid[
            self.amplitude_bounds[0]:self.amplitude_bounds[1]:self.block_sizes[0],
            self.phase_bounds[0]:self.phase_bounds[1]:self.block_sizes[1]
            ]

        parameter_space_tuples = np.vstack((amplitude_spectrum.flatten(), phase_spectrum.flatten())).T

        fixed_validation_tasks = []

        def generate_sin(amplitude, phase):
            def modified_sin(x):
                return amplitude * np.sin(phase * x)
            return modified_sin

        for param_pair in parameter_space_tuples:
            fixed_validation_tasks.append(generate_sin(amplitude=param_pair[0], phase=param_pair[1]))

        return fixed_validation_tasks

    def _compute_loss(self, prediction, ground_truth):
        loss_function = nn.MSELoss()
        return loss_function(prediction, ground_truth)


class _SinusoidalNetwork(ModelNetwork):

    # Make this a more general regression class rather than a Sin specific class?
    def __init__(self, params):
        ModelNetwork.__init__(self, params)

    def construct_layers(self):
        self.linear1 = nn.Linear(self.params.get("x_dim"), 40)
        self.linear2 = nn.Linear(40, 40)
        self.linear3 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SinusoidalNetwork(ModelNetwork):

    # Make this a more general regression class rather than a Sin specific class?
    def __init__(self, params):
        ModelNetwork.__init__(self, params)

    def _construct_layers(self):

        for l in range(len(self.layer_dimensions) - 1):

            layer_weight_tensor = torch.Tensor(size=(self.layer_dimensions[l], self.layer_dimensions[l + 1])).to(self.device)
            layer_weight_tensor.requires_grad = True

            layer_bias_tensor = torch.Tensor(size=[self.layer_dimensions[l + 1]]).to(self.device)
            layer_bias_tensor.requires_grad = True

            self.weights.append(layer_weight_tensor)
            self.biases.append(layer_bias_tensor)

        self._reset_parameters()

    def forward(self, x):

        for l in range(len(self.weights) - 1):
            x = F.linear(x, self.weights[l].t(), self.biases[l])
            x = F.relu(x)
    
        y = F.linear(x, self.weights[-1].t(), self.biases[-1]) # no relu on output layer

        return y