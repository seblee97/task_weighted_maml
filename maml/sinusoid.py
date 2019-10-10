from .model import MAML, ModelNetwork
from utils.priority import PriorityQueue

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Tuple

class SineMAML(MAML):

    def __init__(self, params, device):
        self.device = device
        self.task_type = params.get('task_type')

        # extract relevant task-specific parameters
        self.amplitude_bounds = params.get(['sin2d', 'amplitude_bounds'])
        self.domain_bounds = params.get(['sin2d', 'domain_bounds'])
        degree_phase_bounds = params.get(['sin2d', 'phase_bounds']) # phase given in degrees
        
        if self.task_type == 'sin3d':
            self.frequency_bounds = params.get(['sin3d', 'frequency_bounds'])
            block_sizes = params.get(['sin3d', 'fixed_val_blocks']) 
        else:
            block_sizes = params.get(['sin2d', 'fixed_val_blocks'])

        # convert phase bounds/ fixed_val_interval from degrees to radians
        self.phase_bounds = [
            degree_phase_bounds[0] * (2 * np.pi) / 360, degree_phase_bounds[1] * (2 * np.pi) / 360
            ]
        
        block_sizes[1] = block_sizes[1] * (2 * np.pi) / 360
        self.validation_block_sizes = block_sizes

        self.is_classification = False

        self.model_inner = SinusoidalNetwork(params).to(self.device)

        MAML.__init__(self, params)

    def _get_priority_queue(self):
        """Initiate priority queue"""
        if self.task_type == 'sin3d':
            param_ranges = self.params.get(["priority_queue", "param_ranges_3d"])
            block_sizes = self.params.get(["priority_queue", "block_sizes_3d"])
        elif self.task_type == 'sin2d':
            param_ranges = self.params.get(["priority_queue", "param_ranges_2d"])
            block_sizes = self.params.get(["priority_queue", "block_sizes_2d"])
        return  SinePriorityQueue(
                    queue_resume=self.params.get(["resume", "priority_queue"]),
                    counts_resume=self.params.get(["resume", "queue_counts"]),
                    sample_type=self.params.get(["priority_queue", "sample_type"]),
                    block_sizes=block_sizes,
                    param_ranges=param_ranges,
                    initial_value=self.params.get(["priority_queue", "initial_value"]),
                    epsilon_start=self.params.get(["priority_queue", "epsilon_start"]),
                    epsilon_final=self.params.get(["priority_queue", "epsilon_final"]),
                    epsilon_decay_start=self.params.get(["priority_queue", "epsilon_decay_start"]),
                    epsilon_decay_rate=self.params.get(["priority_queue", "epsilon_decay_rate"]),
                    burn_in=self.params.get(["priority_queue", "burn_in"]),
                    save_path=self.checkpoint_path
                    )

    def _sample_task(self, plot=False):
        """
        returns sin function squashed in x direction by a phase parameter sampled randomly between phase_bounds
        enlarged in the y direction by an apmplitude parameter sampled randomly between amplitude_bounds
        """
        amplitude = random.uniform(self.amplitude_bounds[0], self.amplitude_bounds[1])
        phase = random.uniform(self.phase_bounds[0], self.phase_bounds[1])
        def modified_sin(x):
            return amplitude * np.sin(phase + x)
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
            return parameters[0] * np.sin(parameters[1] + x)
        return modified_sin

    def visualise(self, model_iterations, task, validation_x, validation_y, save_name, visualise_all=True):

        dummy_model = SinusoidalNetwork(self.params)

        # ground truth
        plot_x = np.linspace(self.domain_bounds[0], self.domain_bounds[1], 100)
        plot_x_tensor = torch.tensor([[x] for x in plot_x]).to(self.device)
        plot_y_ground_truth = [task(xi) for xi in plot_x]

        fig = plt.figure()
        plt.plot(plot_x, plot_y_ground_truth, label="Ground Truth")

        dummy_model.weights = model_iterations[-1][0]
        dummy_model.biases = model_iterations[-1][1]

        final_plot_y_prediction = dummy_model(plot_x_tensor)
        plt.plot(plot_x, final_plot_y_prediction.cpu().detach().numpy(), linestyle='dashed', linewidth=3.0, label='Fine-tuned MAML final update')

        if visualise_all:
            for i, (model_weights, model_biases) in enumerate(model_iterations[:-1]):
                
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
        if self.task_type == 'sin3d':
            amplitude_spectrum, phase_spectrum, frequency_spectrum = np.mgrid[
                self.amplitude_bounds[0]:self.amplitude_bounds[1]:self.validation_block_sizes[0],
                self.phase_bounds[0]:self.phase_bounds[1]:self.validation_block_sizes[1],
                self.frequency_bounds[0]:self.frequency_bounds[1]:self.validation_block_sizes[2]
                ]
            parameter_space_tuples = np.vstack((amplitude_spectrum.flatten(), phase_spectrum.flatten(), frequency_spectrum.flatten())).T
        else:
            amplitude_spectrum, phase_spectrum = np.mgrid[
                self.amplitude_bounds[0]:self.amplitude_bounds[1]:self.validation_block_sizes[0],
                self.phase_bounds[0]:self.phase_bounds[1]:self.validation_block_sizes[1]
                ]
            parameter_space_tuples = np.vstack((amplitude_spectrum.flatten(), phase_spectrum.flatten())).T

        fixed_validation_tasks = []

        def generate_sin(amplitude, phase, frequency=1):
            def modified_sin(x):
                return amplitude * np.sin(phase + frequency * x)
            return modified_sin

        for param_pair in parameter_space_tuples:
            if self.task_type == 'sin3d':
                fixed_validation_tasks.append(generate_sin(amplitude=param_pair[0], phase=param_pair[1], frequency=param_pair[2]))
            else:
                fixed_validation_tasks.append(generate_sin(amplitude=param_pair[0], phase=param_pair[1]))

        return parameter_space_tuples, fixed_validation_tasks

    def _compute_loss(self, prediction, ground_truth):
        loss_function = nn.MSELoss()
        return loss_function(prediction, ground_truth)


class _SinusoidalNetwork(ModelNetwork):

    # Make this a more general regression class rather than a Sin specific class?
    def __init__(self, params):
        ModelNetwork.__init__(self, params)

    def _construct_layers(self):
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

class SinePriorityQueue(PriorityQueue):

    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: Dict[str, Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float, epsilon_decay_start: int,
                queue_resume: str, counts_resume: str, save_path: str, burn_in: int=None, initial_value: float=None
                ):

        # convert phase bounds/ phase block_size from degrees to radians
        phase_ranges = [
            param_ranges[1][0] * (2 * np.pi) / 360, param_ranges[1][1] * (2 * np.pi) / 360
            ]
        phase_block_size = block_sizes[1] * (2 * np.pi) / 360

        param_ranges = [param_ranges[0], phase_ranges]
        block_sizes = [block_sizes[0], phase_block_size]
        
        super().__init__(
            block_sizes=block_sizes, param_ranges=param_ranges, sample_type=sample_type, epsilon_start=epsilon_start,
            epsilon_final=epsilon_final, epsilon_decay_rate=epsilon_decay_rate, epsilon_decay_start=epsilon_decay_start, queue_resume=queue_resume,
            counts_resume=counts_resume, save_path=save_path, burn_in=burn_in, initial_value=initial_value
        )

        self.figure_locsx, self.figure_locsy, self.figure_labelsx, self.figure_labelsy = self._get_figure_labels()

    def _get_figure_labels(self):
        xlocs = np.arange(0, self._queue.shape[1])
        ylocs = np.arange(0, self._queue.shape[0])
        xlabels = np.arange(self.param_ranges[1][0], self.param_ranges[1][1], self.block_sizes[1])
        ylabels = np.arange(self.param_ranges[0][0], self.param_ranges[0][1], self.block_sizes[0])
        return xlocs, ylocs, xlabels, ylabels

    def visualise_priority_queue(self, feature='losses'):
        """
        Produces plot of priority queue (losses or counts) 

        Discrete vs continuous, 2d heatmap vs 3d.

        :param feature: which aspect of queue to visualise. 'losses' or 'counts'
        :retrun fig: matplotlib figure showing heatmap of priority queue feature
        """
        if type(self._queue) == np.ndarray:
            if len(self._queue.shape) == 2:
                fig = plt.figure()
                if feature == 'losses':
                    plt.imshow(self._queue)
                elif feature == 'counts':
                    plt.imshow(self.sample_counts)
                else:
                    raise ValueError("feature type not recognised. Use 'losses' or 'counts'")
                plt.colorbar()
                plt.xlabel("Phase")
                plt.ylabel("Amplitude")

                # set labels to sine specific parameter ranges
                # plt.xticks(
                #     locs=self.figure_locsx, 
                #     labels=self.figure_labelsx
                #     )
                # plt.yticks(
                #     locs=self.figure_locsy, 
                #     labels=self.figure_labelsy
                #     )

                return fig
            else:
                warnings.warn("Visualisation with parameter space dimension > 2 not supported", Warning)   
                return None
        else:
            raise NotImplementedError("Visualisation for dictionary queue not implemented")

    def visualise_priority_queue_loss_distribution(self):
        """
        Produces probability distribution plot of losses in the priority queue
        """
        all_losses = self._queue.flatten()

        hist, bin_edges = np.histogram(all_losses, bins=int(0.1 * len(all_losses)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = plt.figure()
        plt.plot(bin_centers, hist)
        return fig

    def visualise_sample_counts(self):
        """
        Produces plot of priority queue sampling counts 
        """
        fig = plt.figure()
        plt.imshow(self.sample_counts)
        plt.colorbar()
        plt.xlabel("Phase")
        plt.ylabel("Amplitude")

        # set labels to sine specific parameter ranges
        # plt.xticks(
        #     locs=self.figure_locsx, 
        #     labels=self.figure_labelsx
        #     )
        # plt.yticks(
        #     locs=self.figure_locsy, 
        #     labels=self.figure_labelsy
        #     )

        return fig