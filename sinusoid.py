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

    def __init__(self, params, device):
        self.device = device

        self.model_inner = SinusoidalNetwork(params).to(self.device)
        # load previously trained model to continue with
        if params.get("resume"):
            model_checkpoint = params.get("resume")
            print("Loading and resuming training from checkpoint @ {}".format(model_checkpoint))
            self.model_inner.load_state_dict(torch.load(model_checkpoint))

        MAML.__init__(self, params)

    def _sample_task(self, amplitude_bounds=(0.1, 5), phase_bounds=(0, math.pi), plot=False):
        """
        returns sin function squashed in x direction by a phase parameter sampled randomly between phase_bounds
        enlarged in the y direction by an apmplitude parameter sampled randomly between amplitude_bounds
        """
        amplitude = random.uniform(amplitude_bounds[0], amplitude_bounds[1])
        phase = random.uniform(phase_bounds[0], phase_bounds[1])
        def modified_sin(x):
            return amplitude * np.sin(phase * x)
        return modified_sin

    def visualise(self, model_iterations, task, save_name, domain_bounds=(-5, 5)):

        dummy_model = SinusoidalNetwork(self.params)

        # ground truth
        plot_x = np.linspace(domain_bounds[0], domain_bounds[1], 100)
        plot_x_tensor = torch.tensor([[x] for x in plot_x]).to(self.device)
        plot_y_ground_truth = [task(xi) for xi in plot_x]

        fig = plt.figure()
        plt.plot(plot_x, plot_y_ground_truth)

        for (model_weights, model_biases) in model_iterations:
            
            dummy_model.weights = model_weights
            dummy_model.biases = model_biases

            plot_y_prediction = dummy_model(plot_x_tensor)
            plt.plot(plot_x, plot_y_prediction.cpu().detach().numpy(), linestyle='dashed')
        # plt.scatter(test_x_batch.cpu(), test_y_batch.cpu(), marker='o')
        fig.savefig(self.params.get("checkpoint_path") + save_name)
        plt.close()

    def _generate_batch(self, task, domain_bounds=(-5, 5), batch_size=10, plot=False): # Change batch generation to be done in pure PyTorch
        """
        generates an array, x_batch, of B datapoints sampled randomly between domain_bounds
        and computes the sin of each point in x_batch to produce y_batch.
        """
        x_batch = [random.uniform(domain_bounds[0], domain_bounds[1]) for _ in range(batch_size)]
        y_batch = [task(x) for x in x_batch]
        x_batch_tensor = torch.tensor([[x] for x in x_batch]).to(self.device)
        y_batch_tensor = torch.tensor([[y] for y in y_batch]).to(self.device)
        # print(x_batch_tensor, y_batch_tensor)
        # print(x_batch_tensor.shape, y_batch_tensor.shape)
        # print(x_batch_tensor.dtype)       

        if plot:
            fig = plt.figure()
            x = np.linspace(domain_bounds[0], domain_bounds[1], 100)
            y = [task(xi) for xi in x]
            plt.plot(x, y)
            fig.savefig('sin_batch_test.png')
            plt.close()
        return x_batch_tensor, y_batch_tensor

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