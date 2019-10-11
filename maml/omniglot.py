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
        x_batch_tensor = torch.Tensor(x_batch.reshape(x_tensor_shape))

        y_batch_tensor = torch.Tensor(unstacked_y_batch).to(torch.long)

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

        self.weight_layer_keys = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'linear.weight']
        self.bias_layer_keys = ['conv1.bias', 'conv2.bias', 'conv3.bias', 'conv4.bias', 'linear.bias']

    def _construct_layers(self):
        network_layers = self.params.get("network_layers")
        in_channels = self.params.get(["omniglot", "in_channels"])
        filter_size = self.params.get(["omniglot", "filter_size"])
        output_dimension = self.params.get(["omniglot", "N"])
        input_dimension = self.params.get(["omniglot", "image_output_shape"])
        stride_mag = self.params.get(["omniglot", "stride"])
        padding_mag = self.params.get(["omniglot", "padding"])

        self.conv1 = nn.Conv2d(in_channels, out_channels=network_layers[0], kernel_size=filter_size, stride=stride_mag, padding=padding_mag)
        self.conv2 = nn.Conv2d(network_layers[0], out_channels=network_layers[1], kernel_size=filter_size, stride=stride_mag, padding=padding_mag)
        self.conv3 = nn.Conv2d(network_layers[1], out_channels=network_layers[2], kernel_size=filter_size, stride=stride_mag, padding=padding_mag)
        self.conv4 = nn.Conv2d(network_layers[2], out_channels=network_layers[3], kernel_size=filter_size, stride=stride_mag, padding=padding_mag)
        
        self.bn1 = nn.BatchNorm2d(network_layers[0])
        self.bn2 = nn.BatchNorm2d(network_layers[1])
        self.bn3 = nn.BatchNorm2d(network_layers[2])
        self.bn4 = nn.BatchNorm2d(network_layers[3])

        flattened_dimension = get_convolutional_output_dims(
            input_shape=input_dimension, output_depth=network_layers[-1], kernel_sizes=[filter_size for _ in range(len(network_layers))], 
            strides=[stride_mag for _ in range(len(network_layers))], paddings=[padding_mag for _ in range(len(network_layers))]
        )

        flattened_dimension = 256

        self.linear = nn.Linear(flattened_dimension, output_dimension)

        # self._reset_parameters()

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.reshape((25, -1))
        print(x.shape)

        x = F.softmax(self.linear(x))

        return x

    def get_weights(self):
        return [self.state_dict()[l] for l in self.weight_layer_keys]

    def get_biases(self):
        return [self.state_dict()[l] for l in self.bias_layer_keys]

    def set_weight_gradients(self, layer_index, gradient):
        self.state_dict()[self.weight_layer_keys[layer_index]].grad = gradient

    def set_bias_gradients(self, layer_index, gradient):
        self.state_dict()[self.bias_layer_keys[layer_index]].grad = gradient

    def set_weights(self, weights: List):
        for l, weight_layer in enumerate(weights):
            self.state_dict()[self.weight_layer_keys[l]] = weight_layer

    def set_biases(self, biases: List):
        for l, bias_layer in enumerate(biases):
            self.state_dict()[self.bias_layer_keys[l]] = bias_layer

    def update_weights(self, layer_index, gradients, learning_rate):
        self.state_dict()[self.weight_layer_keys[layer_index]] -= learning_rate * gradients

    def update_biases(self, layer_index, gradients, learning_rate):
        self.state_dict()[self.bias_layer_keys[layer_index]] -= learning_rate * gradients


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
