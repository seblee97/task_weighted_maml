from .model import MAML, ModelNetwork
from utils.priority import PriorityQueue
from utils.custom_functions import get_convolutional_output_dims

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F

import copy
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import os

from typing import Any, Dict, List, Tuple

class OmniglotMAML(MAML):

    def __init__(self, params, device):
        self.device = device
        self.task_type = params.get('task_type')

        # extract relevant task-specific parameters
        self.k = params.get('inner_update_k')
        self.n_train = params.get(['omniglot', 'n_train'])
        self.examples_per_class = params.get(['omniglot', 'examples_per_class'])
        self.image_shape = params.get(['omniglot', 'image_output_shape'])
        self.N = params.get(['omniglot', 'N'])
        self.batch_size = params.get('task_batch_size')
        self.one_hot_ground_truth = params.get(['omniglot', 'one_hot_ground_truth'])
        
        # load image data
        print("Loading image dataset...")
        t0 = time.time()
        training_data_path = os.path.join(params.get(["omniglot", "data_path"]), "train_data")
        self.training_data = [[np.load(os.path.join(training_data_path, char, instance)) 
                               for instance in os.listdir(os.path.join(training_data_path, char))] for 
                               char in os.listdir(training_data_path)]
        test_data_path = os.path.join(params.get(["omniglot", "data_path"]), "test_data")
        self.test_data = [[np.load(os.path.join(test_data_path, char, instance)) 
                               for instance in os.listdir(os.path.join(test_data_path, char))] for 
                               char in os.listdir(test_data_path)]

        self.data = self.training_data + self.test_data
        self.total_num_classes = len(self.data)
        print("Finished loading image dataset ({} seconds).".format(round(time.time() - t0)))

        self.is_classification = True

        self.model_inner = OmniglotNetwork(params).to(self.device)

        MAML.__init__(self, params)

    def _get_priority_queue(self):
        """Initiate priority queue"""
        param_ranges = self.params.get(["priority_queue", "param_ranges"])
        block_sizes = self.params.get(["priority_queue", "block_sizes"])
        return  OmniglotPriorityQueue(
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
                    save_path=self.checkpoint_path,
                    scale_parameters=self.params.get(["priority_queue", "scale_parameters"])
                    )

    def _sample_task(self, plot=False):
        """
        returns list of indices corresponding to images in dataset
        """
        task = []
        image_count = 0
        while image_count < self.N:
            image_index = random.randrange(self.n_train)
            if image_index in task:
                pass 
            else:
                image_count += 1
                task.append(image_index)
        return task

    def _get_task_from_params(self, parameters: List[float]) -> Any:
        """
        Return sine function defined by parameters given

        :param parameters: parameters defining the specific sin task in the distribution

        :return modified_sin: sin function

        (method differs from _sample_task in that it is not a random sample but
        defined by parameters given)
        """
        raise NotImplementedError

    def visualise(self, model_iterations, task, validation_x, validation_y, save_name, visualise_all=True):
        """
        Visualise qualitative run.

        :param validation_model_iterations: parameters of model after successive fine-tuning steps
        :param val_task: task being evaluated
        :param validation_x_batch: k data points fed to model for finetuning
        :param validation_y_batch: ground truth data associated with validation_x_batch
        :param save_name: name of file to be saved
        :param visualise_all: whether to visualise all fine-tuning steps or just final 
        """

        if visualise_all:
            figure_y_dimension = self.N + len(model_iterations)
        else:
            figure_y_dimension = self.N + 1

        fig = plt.figure(figsize=(self.k, figure_y_dimension))
        fig.suptitle("Input to N-way, k-shot Classification")

        grid_spec = gridspec.GridSpec(
            figure_y_dimension, 
            self.k, 
            figure=fig, 
            height_ratios=[1 for _ in range(self.N)] + [3 for _ in range(figure_y_dimension - self.N)], 
            width_ratios=[1 for _ in range(self.k)]
            )

        nk_validation_x = validation_x.reshape(self.N, self.k, self.image_shape[0], self.image_shape[1])
        nk_validation_y = validation_y.reshape(self.N, self.k)
        
        # add input data to figure
        for i in range(self.N):
            for j in range(self.k):
                ax = fig.add_subplot(grid_spec[i, j])
                ax.imshow(nk_validation_x[i][j])
                ax.set_title("Class {} out of {}".format(i, self.N), fontdict={'fontsize':5})
                ax.set_xticks([])
                ax.set_yticks([])

        dummy_model = OmniglotNetwork(self.params)

        dummy_model._weights = model_iterations[-1][0]
        dummy_model._biases = model_iterations[-1][1]

        final_predictions_array = dummy_model(validation_x)
        final_predictions_accuracy, final_accuracy_matrix = self._get_accuracy(logits=final_predictions_array, ground_truth=validation_y, return_plot=True)

        final_prediction_axis = fig.add_subplot(grid_spec[self.N, :])
        final_prediction_axis.imshow(final_accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        final_prediction_axis.set_title("Accuracy: {}".format(final_predictions_accuracy))

        if visualise_all:
            for i, (model_weights, model_biases) in enumerate(model_iterations[:-1]):
                
                dummy_model._weights = model_weights
                dummy_model._biases = model_biases

                y_prediction_array = dummy_model(validation_x)
                y_prediction_accuracy, accuracy_matrix = self._get_accuracy(logits=y_prediction_array, ground_truth=validation_y, return_plot=True)

                prediction_axis = fig.add_subplot(grid_spec[self.N + 1 + i, :])
                prediction_axis.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        
        # fig.savefig(self.params.get("checkpoint_path") + save_name)

        plt.close()

        return fig

    def _generate_batch(self, task, batch_size=10, plot=False): # Change batch generation to be done in pure PyTorch
        """
        generates an array, x_batch, of B datapoints sampled randomly between domain_bounds
        and computes the sin of each point in x_batch to produce y_batch.
        """
        unstacked_x_batch = [[self.data[n][example] for example in random.sample(list(range(self.examples_per_class)), self.k)] for n in task]
        # print(x_batch_tensor, y_batch_tensor)
        # print(x_batch_tensor.shape, y_batch_tensor.shape)
        # print(x_batch_tensor.dtype)

        if self.one_hot_ground_truth:
            batch_y_entry = np.zeros((self.k * self.N, self.N))
            ground_truth_indices = np.concatenate([[j for _ in range(self.k)] for j in range(self.N)])
            batch_y_entry[range(self.k * self.N), ground_truth_indices] = 1
        else:
            unstacked_y_batch = np.concatenate([[j for _ in range(self.k)] for j in range(self.N)])
        
        x_batch = np.expand_dims(np.stack(unstacked_x_batch), axis=-1)
        x_tensor_shape = (self.N * self.k, 1, self.image_shape[0], self.image_shape[1])
        x_batch_tensor = torch.Tensor(x_batch.reshape(x_tensor_shape)).to(self.device)

        y_batch_tensor = torch.Tensor(unstacked_y_batch).to(torch.long).to(self.device)

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
        loss_function = nn.CrossEntropyLoss()
        return loss_function(prediction, ground_truth)


class OmniglotNetwork(ModelNetwork):

    # Make this a more general regression class rather than a Sin specific class?
    def __init__(self, params):
        ModelNetwork.__init__(self, params)

        self.stride_mag = self.params.get(["omniglot", "stride"])
        self.padding_mag = self.params.get(["omniglot", "padding"])

    def _construct_layers(self):
        network_layers = self.params.get("network_layers")
        in_channels = self.params.get(["omniglot", "in_channels"])
        filter_size = self.params.get(["omniglot", "filter_size"])
        output_dimension = self.params.get(["omniglot", "N"])
        input_dimension = self.params.get(["omniglot", "image_output_shape"])
        channels = [in_channels] + network_layers[:-1]

        for l in range(len(network_layers)):
            
            # convolutional layers
            layer_weight_tensor = torch.Tensor(size=(network_layers[l], channels[l], filter_size, filter_size)).to(self.device)
            layer_weight_tensor.requires_grad = True

            layer_bias_tensor = torch.Tensor(size=[network_layers[l]]).to(self.device)
            layer_bias_tensor.requires_grad = True

            self._weights.append(layer_weight_tensor)
            self._biases.append(layer_bias_tensor)

            # batch norm layers
            batch_norm_weight_size = [network_layers[l]]
            batch_norm_bias_size = [network_layers[l]]

            batch_norm_weight_tensor = torch.Tensor(size=batch_norm_weight_size)
            batch_norm_weight_tensor.requires_grad = True
            
            batch_norm_bias_tensor = torch.Tensor(size=batch_norm_bias_size)
            batch_norm_bias_tensor.requires_grad = True

            self._weights.append(batch_norm_weight_tensor)
            self._biases.append(batch_norm_bias_tensor)

        flattened_dimension = 256

        linear_layer_weight_tensor = torch.Tensor(size=(flattened_dimension, output_dimension)).to(self.device)
        linear_layer_weight_tensor.requires_grad = True

        linear_layer_bias_tensor = torch.Tensor(size=[output_dimension])
        linear_layer_bias_tensor.requires_grad = True

        self._weights.append(linear_layer_weight_tensor)
        self._biases.append(linear_layer_bias_tensor)

        self._reset_parameters()

    def forward(self, x):

        for l in range(0, len(self._weights) - 1, 2):
            x = F.conv2d(x, self._weights[l], self._biases[l], stride=self.stride_mag, padding=self.padding_mag)
            
            input_mean = torch.mean(x, dim=[0, 2, 3]).to(self.device).detach()
            input_var = torch.var(x, dim=[0, 2, 3]).to(self.device).detach()
            
            x = F.batch_norm(
                x, running_mean=input_mean, running_var=input_var, weight=self._weights[l + 1], bias=self._biases[l + 1]
                )
            x = F.relu(x)

        x = x.reshape((25, -1))
        x = F.softmax(F.linear(x, self._weights[-1].t(), self._biases[-1]), dim=1)

        return x


class OmniglotPriorityQueue(PriorityQueue):

    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: List[Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float, epsilon_decay_start: int,
                queue_resume: str, counts_resume: str, save_path: str, burn_in: int=None, initial_value: float=None, scale_parameters: bool=False
                ):
        
        super().__init__(
            block_sizes=block_sizes, param_ranges=param_ranges, sample_type=sample_type, epsilon_start=epsilon_start,
            epsilon_final=epsilon_final, epsilon_decay_rate=epsilon_decay_rate, epsilon_decay_start=epsilon_decay_start, queue_resume=queue_resume,
            counts_resume=counts_resume, save_path=save_path, burn_in=burn_in, initial_value=initial_value, scale_parameters=scale_parameters
        )

    def visualise_priority_queue(self, feature='losses'):
        """
        Produces plot of priority queue (losses or counts) 

        Discrete vs continuous, 2d heatmap vs 3d.
    
        :param feature: which aspect of queue to visualise. 'losses' or 'counts'
        :retrun fig: matplotlib figure showing heatmap of priority queue feature
        """
        def closestDivisors(N):
            """Finds two highest factors that are closest to each other"""
            first_approx = round(np.sqrt(N))
            while N % first_approx > 0:
                first_approx -= 1
            return (int(first_approx), int(N / first_approx))

        if type(self._queue) == np.ndarray:
            if len(self._queue.shape) <= 2:

                if feature == 'losses':
                    plot_queue = copy.deepcopy(self._queue)
                elif feature == 'counts':
                    plot_queue = copy.deepcopy(sample_counts)
                else:
                    raise ValueError("feature type not recognised. Use 'losses' or 'counts'")

                if len(self._queue.shape) == 1:
                    plot_queue = plot_queue.reshape((len(plot_queue), 1))

                # re-order queue to sorted by loss (makes visual more interpretable)
                plot_queue = plot_queue[np.argsort(self._queue)]

                # reshape queue to something visually tractable
                plot_queue = plot_queue.reshape(closestDivisors(len(plot_queue)))

                fig = plt.figure()
                plt.imshow(plot_queue)                
                plt.colorbar()
                plt.xlabel("Image Index")

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
