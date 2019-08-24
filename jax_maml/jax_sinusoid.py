from .jax_model import MAML
from utils.priority import PriorityQueue

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Tuple

from jax.experimental import stax # neural network library
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax

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

        MAML.__init__(self, params)

    def _get_priority_queue(self):
        return  SinePriorityQueue(
                    queue_resume=self.params.get(["resume", "priority_queue"]),
                    counts_resume=self.params.get(["resume", "queue_counts"]),
                    sample_type=self.params.get(["priority_queue", "sample_type"]),
                    block_sizes=self.params.get(["priority_queue", "block_sizes"]),
                    param_ranges=self.params.get(["priority_queue", "param_ranges"]),
                    initial_value=self.params.get(["priority_queue", "initial_value"]),
                    epsilon_start=self.params.get(["priority_queue", "epsilon_start"]),
                    epsilon_final=self.params.get(["priority_queue", "epsilon_final"]),
                    epsilon_decay_start=self.params.get(["priority_queue", "epsilon_decay_start"]),
                    epsilon_decay_rate=self.params.get(["priority_queue", "epsilon_decay_rate"]),
                    burn_in=self.params.get(["priority_queue", "burn_in"]),
                    save_path=self.checkpoint_path
                    )

    def _get_model(self):
        return stax.serial(
            Dense(40), Relu,
            Dense(40), Relu,
            Dense(1)
            )
    
    def _get_optimiser(self):
        return optimizers.adam(step_size=self.meta_lr)

    def _sample_task(self, plot=False):
        """
        returns sin function squashed in x direction by a phase parameter sampled randomly between phase_bounds
        enlarged in the y direction by an apmplitude parameter sampled randomly between amplitude_bounds
        """
        tasks = []
        for _ in range(self.task_batch_size):
            amplitude = random.uniform(self.amplitude_bounds[0], self.amplitude_bounds[1])
            phase = random.uniform(self.phase_bounds[0], self.phase_bounds[1])
            def modified_sin(x):
                return amplitude * np.sin(phase + x)
            tasks.append(modified_sin)
        return tasks

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

    def _generate_batch(self, tasks: List, plot=False): # Change batch generation to be done in pure PyTorch
        """
        generates an array, x_batch, of B datapoints sampled randomly between domain_bounds
        and computes the sin of each point in x_batch to produce y_batch.
        """
        x_batch = np.stack([np.random.uniform(low=self.domain_bounds[0], high=self.domain_bounds[1], size=(self.inner_update_k, 1)) for _ in range(len(tasks))])
        y_batch = np.stack([[tasks[t](x) for x in x_batch[t]] for t in range(len(tasks))])

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
        return x_batch, y_batch

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

        return parameter_space_tuples, fixed_validation_tasks

    def _compute_loss(self, parameters, inputs, ground_truth):
        predictions = self.network_forward(parameters, inputs)
        return np.mean((ground_truth - predictions) ** 2)


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
        xlocs = np.arange(0, self.queue.shape[1])
        ylocs = np.arange(0, self.queue.shape[0])
        xlabels = np.arange(self.param_ranges[1][0], self.param_ranges[1][1], self.block_sizes[1])
        ylabels = np.arange(self.param_ranges[0][0], self.param_ranges[0][1], self.block_sizes[0])
        return xlocs, ylocs, xlabels, ylabels

    def visualise_priority_queue(self):
        """
        Produces plot of priority queue. 

        Discrete vs continuous, 2d heatmap vs 3d.
        """
        if type(self.queue) == np.ndarray:
            if len(self.queue.shape) == 2:
                fig = plt.figure()
                plt.imshow(self.queue)
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
                raise ValueError("Visualisation with parameter space dimension > 2 not supported")
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