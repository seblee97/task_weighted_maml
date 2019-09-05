from .jax_model import MAML
from utils.priority import PriorityQueue

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings

from typing import Any, Dict, List, Tuple

from jax.experimental import stax # neural network library
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax

class RLMAML(MAML):

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

        MAML.__init__(self, params)

    def _get_priority_queue(self):
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

    def _get_model(self):
        """
        Returns policy network
        """
        layers = []

        # inner / hidden network layers + non-linearities
        for l in self.network_layers:
            layers.append(Dense(l))
            layers.append(Relu)

        # output layer (no non-linearity)
        layers.append(Dense(self.output_dsimension))
        
        return stax.serial(*layers)

        raise NotImplementedError
    def __init__(self, input_size, output_size, hidden_sizes=(),
                 nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        super(NormalMLPPolicy, self).__init__(input_size=input_size, output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = math.log(min_std)
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)

        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(init_std))
        self.apply(weight_init)

    def forward(self, input, params=None):

        if params is None:
            params = OrderedDict(self.named_parameters())

        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'],
                      bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))

        return Normal(loc=mu, scale=scale)
    
    def _get_optimiser(self):
        return optimizers.adam(step_size=self.meta_lr)

    def _sample_task(self, batch_size, plot=False, validate=False, step_count=None):
        """
        returns batch of RL gym environments defined according to the parameters specific to this RL task.
        """
        raise NotImplementedError

        tasks = []
        all_max_indices = [] if self.priority_sample else None

        for _ in range(batch_size):

            # sample a task from task distribution and generate x, y tensors for that task
            if self.priority_sample and not validate:

                # query queue for next task parameters
                max_indices, task_parameters = self.priority_queue.query(step=step_count)
                all_max_indices.append(max_indices)

                # get epsilon value
                epsilon = self.priority_queue.get_epsilon()

                # get task from parameters returned from query
                amplitude = task_parameters[0]
                phase = task_parameters[1]
                if self.task_type == 'sin3d':
                    frequency_scaling = task_parameters[2]
                else:
                    frequency_scaling = 1.
                task = self._get_task_from_params(amplitude=amplitude, phase=phase, frequency_scaling=frequency_scaling)

                # compute metrics for tb logging
                queue_count_loss_correlation = self.priority_queue.compute_count_loss_correlation()
                queue_mean = np.mean(self.priority_queue.get_queue())
                queue_std = np.std(self.priority_queue.get_queue())

                # write to tensorboard
                if epsilon:
                    self.writer.add_scalar('queue_metrics/epsilon', epsilon, step_count)
                self.writer.add_scalar('queue_metrics/queue_correlation', queue_count_loss_correlation, step_count)
                self.writer.add_scalar('queue_metrics/queue_mean', queue_mean, step_count)
                self.writer.add_scalar('queue_metrics/queue_std', queue_std, step_count)

            else:
                
                # sample randomly (vanilla maml)
                amplitude = random.uniform(self.amplitude_bounds[0], self.amplitude_bounds[1])
                phase = random.uniform(self.phase_bounds[0], self.phase_bounds[1])
                
                if self.task_type == 'sin3d':
                    frequency_scaling = random.uniform(self.frequency_bounds[0], self.frequency_bounds[1])
                else:
                    frequency_scaling = 1.
                
                task = self._get_task_from_params(amplitude=amplitude, phase=phase, frequency_scaling=frequency_scaling)
                
            tasks.append(task)
    
        return tasks, all_max_indices

    def _get_task_from_params(self, amplitude: float, phase: float, frequency_scaling: float=1.) -> Any:
        """
        Return sine function defined by parameters given

        :param parameters: parameters defining the specific sin task in the distribution

        :return modified_sin: sin function

        (method differs from _sample_task in that it is not a random sample but
        defined by parameters given)
        """
        def modified_sin(x):
            return amplitude * np.sin(phase + frequency_scaling * x)
        return modified_sin

    def visualise(self, model_iterations, task, validation_x, validation_y, save_name, visualise_all=True):
        """
        Gif of episode rollout?
        """
        raise NotImplementedError

    def _generate_batch(self, tasks: List, plot=False): # Change batch generation to be done in pure PyTorch
        """
        Samples trajectories for set of RL tasks provided. 
        """
        raise NotImplementedError

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

    def _compute_loss(self, parameters, inputs, ground_truth):
        """
        Computes loss of network

        :param parameters: current weights of model
        :param inputs: x data
        :param ground_truth: y_data

        :return loss: loss on ground truth vs output of network applied to inputs
        """
        predictions = self.network_forward(parameters, inputs)
        loss = np.mean((ground_truth - predictions) ** 2)
        return loss

class SinePriorityQueue(PriorityQueue):

    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: List[Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float, epsilon_decay_start: int,
                queue_resume: str, counts_resume: str, save_path: str, burn_in: int=None, initial_value: float=None
                ):

        # convert phase bounds/ phase block_size from degrees to radians
        phase_ranges = [
            param_ranges[1][0] * (2 * np.pi) / 360, param_ranges[1][1] * (2 * np.pi) / 360
            ]
        phase_block_size = block_sizes[1] * (2 * np.pi) / 360

        param_ranges[1] = phase_ranges
        block_sizes[1] = phase_block_size
        
        super().__init__(
            block_sizes=block_sizes, param_ranges=param_ranges, sample_type=sample_type, epsilon_start=epsilon_start,
            epsilon_final=epsilon_final, epsilon_decay_rate=epsilon_decay_rate, epsilon_decay_start=epsilon_decay_start, queue_resume=queue_resume,
            counts_resume=counts_resume, save_path=save_path, burn_in=burn_in, initial_value=initial_value
        )

        self.figure_locsx, self.figure_locsy, self.figure_labelsx, self.figure_labelsy = self._get_figure_labels()

    def _get_figure_labels(self):
        xlocs = np.arange(0, self.queue.shape[1])
        ylocs = np.arange(0, self.queue.shape[0])
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
        if type(self.queue) == np.ndarray:
            if len(self.queue.shape) == 2:
                fig = plt.figure()
                if feature == 'losses':
                    plt.imshow(self.queue)
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
        all_losses = self.queue.flatten()

        hist, bin_edges = np.histogram(all_losses, bins=int(0.1 * len(all_losses)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = plt.figure()
        plt.plot(bin_centers, hist)
        return fig
