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

        MAML.__init__(self, params)

    def _get_model(self):
        """
        Return jax network initialisation and forward method.
        """
        layers = []

        # inner / hidden network layers + non-linearities
        for l in self.network_layers:
            layers.append(Dense(l))
            layers.append(Relu)

        # output layer (no non-linearity)
        layers.append(Dense(self.output_dimension))
        
        # make jax stax object
        model = stax.serial(*layers)

        return model
    
    def _get_optimiser(self):
        """
        Return jax optimiser: initialisation, update method and parameter getter method.
        Optimiser learning rate is given by config (meta_lr).
        """
        return optimizers.adam(step_size=self.meta_lr)

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

    def _sample_task(self, batch_size, validate=False, step_count=None):
        """
        Sample specific task(s) from defined distribution of tasks 
        E.g. one specific sine function from family of sines

        :param batch_size: number of tasks to sample
        :param validate: whether or not tasks are being used for validation
        :param step_count: step count during training 

        :return tasks: batch of tasks
        :return task_indices: indices of priority queue associated with batch of tasks
        :return task_probabilities: probabilities of tasks sampled being chosen a priori

        Returns batch of sin functions shifted in x direction by a phase parameter sampled randomly between phase_bounds
        (set by config) enlarged in the y direction by an amplitude parameter sampled randomly between amplitude_bounds
        (also set by config). For 3d sine option, function is also squeezed in x direction by freuency parameter.
        """
        tasks = []
        task_probabilities = []
        all_max_indices = [] if self.priority_sample else None

        for _ in range(batch_size):

            # sample a task from task distribution and generate x, y tensors for that task
            if self.priority_sample and not validate:

                # query queue for next task parameters
                max_indices, task_parameters, task_probability = self.priority_queue.query(step=step_count)
                all_max_indices.append(max_indices)
                task_probabilities.append(task_probability)

                # get epsilon value
                epsilon = self.priority_queue.get_epsilon()

                # get task from parameters returned from query
                amplitude = task_parameters[0]
                phase = task_parameters[1]
                if self.task_type == 'sin3d':
                    frequency_scaling = task_parameters[2]
                else:
                    frequency_scaling = 1.
                parameters = [amplitude, phase, frequency_scaling]
                task = self._get_task_from_params(parameters=parameters)

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
                
                parameters = [amplitude, phase, frequency_scaling]
                task = self._get_task_from_params(parameters=parameters)
                
            tasks.append(task)
    
        return tasks, all_max_indices, task_probabilities

    def _get_task_from_params(self, parameters: List) -> Any:
        """
        Return sine function defined by parameters given

        :param parameters: parameters defining the specific sin task in the distribution

        :return modified_sin: sin function

        (method differs from _sample_task in that it is not a random sample but
        defined by parameters given)
        """
        amplitude = parameters[0]
        phase = parameters[1]
        frequency_scaling = parameters[2]
        def modified_sin(x):
            return amplitude * np.sin(phase + frequency_scaling * x)
        return modified_sin

    def _generate_batch(self, tasks: List): 
        """
        Obtain batch of training examples from a list of tasks

        :param tasks: list of tasks for which data points need to be sampled
        
        :return x_batch: x points sampled from data
        :return y_batch: y points associated with x_batch
        """
        x_batch = np.stack([np.random.uniform(low=self.domain_bounds[0], high=self.domain_bounds[1], size=(self.inner_update_k, 1)) for _ in range(len(tasks))])
        y_batch = np.stack([[tasks[t](x) for x in x_batch[t]] for t in range(len(tasks))])

        return x_batch, y_batch

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

    def _visualise(self, model_iterations, task, validation_x, validation_y, save_name, visualise_all=True):
        """
        Visualise qualitative run.

        :param validation_model_iterations: parameters of model after successive fine-tuning steps
        :param val_task: task being evaluated
        :param validation_x_batch: k data points fed to model for finetuning
        :param validation_y_batch: ground truth data associated with validation_x_batch
        :param save_name: name of file to be saved
        :param visualise_all: whether to visualise all fine-tuning steps or just final 
        """

        # ground truth
        plot_x = np.linspace(self.domain_bounds[0], self.domain_bounds[1], 100)
        plot_y_ground_truth = [task(xi) for xi in plot_x]

        fig = plt.figure()
        plt.plot(plot_x, plot_y_ground_truth, label="Ground Truth")

        final_plot_y_prediction = self.network_forward(model_iterations[-1], plot_x.reshape(len(plot_x), 1))
        plt.plot(plot_x, final_plot_y_prediction, linestyle='dashed', linewidth=3.0, label='Fine-tuned MAML final update')

        no_tuning_y_prediction = self.network_forward(model_iterations[0], plot_x.reshape(len(plot_x), 1))
        plt.plot(plot_x, no_tuning_y_prediction, linestyle='dashed', linewidth=3.0, label='Untuned MAML prediction')
        
        if visualise_all:
            for i, (model_iteration) in enumerate(model_iterations[1:-1]):

                plot_y_prediction = self.network_forward(model_iteration, plot_x.reshape(len(plot_x), 1))
                plt.plot(plot_x, plot_y_prediction, linestyle='dashed') #, label='Fine-tuned MAML {} update'.format(i))

        plt.scatter(validation_x, validation_y, marker='o', label='K Points')

        plt.title("Validation of Sinusoid Meta-Regression")
        plt.xlabel(r"x")
        plt.ylabel(r"sin(x)")
        plt.legend()
        
        # fig.savefig(self.params.get("checkpoint_path") + save_name)
        plt.close()

        return fig

    def _get_fixed_validation_tasks(self):
        """
        If using fixed validation this method returns a set of tasks that are 
        equally spread across the task distribution space.
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
