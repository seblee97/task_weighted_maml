from .jax_model import MAML
from utils.priority import PriorityQueue

import os
import copy
import math
import random
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
import time

from typing import Any, Dict, List, Tuple

from jax.experimental import stax # neural network library
from jax.experimental import optimizers
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, BatchNorm, Softmax

import jax.numpy as np

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
        self.training_data = [[onp.load(os.path.join(training_data_path, char, instance)) 
                               for instance in os.listdir(os.path.join(training_data_path, char))] for 
                               char in os.listdir(training_data_path)]
        test_data_path = os.path.join(params.get(["omniglot", "data_path"]), "test_data")
        self.test_data = [[onp.load(os.path.join(test_data_path, char, instance)) 
                               for instance in os.listdir(os.path.join(test_data_path, char))] for 
                               char in os.listdir(test_data_path)]

        self.data = self.training_data + self.test_data
        self.total_num_classes = len(self.data)
        print("Finished loading image dataset ({} seconds).".format(round(time.time() - t0)))

        self.is_classification = True

        MAML.__init__(self, params)

    def _get_model(self):
        """
        Return jax network initialisation and forward method.
        """
        layers = []

        # inner / hidden network layers + non-linearities
        for l in self.network_layers:
            layers.append(Conv(out_chan=l, filter_shape=(3, 3), padding='SAME', strides=(2,2)))
            layers.append(BatchNorm())
            layers.append(Relu)
        
        layers.append(Flatten)
        layers.append(Dense(self.N))
        layers.append(Softmax)

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

        # if step_count == 1:
        #     import pdb; pdb.set_trace()

        for _ in range(batch_size):

            # sample a task from task distribution and generate x, y tensors for that task
            if self.priority_sample and not validate:

                task_max_indices, task, all_task_probabilities = [], [], []
    
                # query queue for next task parameters
                query_count = 0
                while query_count < self.N:
                    max_indices, task_parameters, task_probability = self.priority_queue.query(step=step_count)
                    if task_parameters[0] in task:
                        pass 
                    else:
                        task_max_indices.append(max_indices[0])
                        task.append(task_parameters[0])
                        all_task_probabilities.append(task_probability)
                        query_count += 1

                all_max_indices.append(task_max_indices)
                task_probabilities.append(onp.prod(all_task_probabilities))

                # get epsilon value
                epsilon = self.priority_queue.get_epsilon()

                # compute metrics for tb logging
                queue_count_loss_correlation = self.priority_queue.compute_count_loss_correlation()
                queue_mean = onp.mean(self.priority_queue.get_queue())
                queue_std = onp.std(self.priority_queue.get_queue())

                # write to tensorboard
                if epsilon:
                    self.writer.add_scalar('queue_metrics/epsilon', epsilon, step_count)
                self.writer.add_scalar('queue_metrics/queue_correlation', queue_count_loss_correlation, step_count)
                self.writer.add_scalar('queue_metrics/queue_mean', queue_mean, step_count)
                self.writer.add_scalar('queue_metrics/queue_std', queue_std, step_count)

            else:
                # sample randomly (vanilla maml)
                if validate:
                    task = [random.randrange(self.n_train, self.total_num_classes) for _ in range(self.N)]
                else:
                    task = [random.randrange(self.n_train) for _ in range(self.N)]
            
            tasks.append(task)
        # import pdb; pdb.set_trace()
    
        return tasks, all_max_indices, task_probabilities

    def _get_task_from_params(self, parameters: List) -> Any:
        """
        Not needed for Image classification tasks

        (method differs from _sample_task in that it is not a random sample but
        defined by parameters given)
        """
        raise NotImplementedError

    def _generate_batch(self, tasks: List[Dict]): 
        """
        Obtain batch of training examples from a list of tasks

        :param tasks: list of tasks for which data points need to be sampled
        
        :return x_batch: x points sampled from data
        :return y_batch: y points associated with x_batch
        """
        unstacked_x_batch = [[[self.data[n][example] for example in random.sample(list(range(self.examples_per_class)), self.k)] for n in task] for task in tasks]
        x_batch = onp.stack(unstacked_x_batch).reshape(-1, self.N * self.k, self.image_shape[0], self.image_shape[1], 1) # B, N*k, width, height, 1

        # one hot vectors where 1 corresponds to correct class
        # unstacked_y_batch = np.zeros((len(unstacked_x_batch), self.n_train))
        # unstacked_y_batch[range(len(unstacked_x_batch)), [task[0] for task in tasks]] = 1
        # y_batch = np.stack(unstacked_y_batch)

        if self.one_hot_ground_truth:
            unstacked_y_batch = []
            for t in range(len(tasks)):
                batch_y_entry = onp.zeros((self.k * self.N, self.N))
                ground_truth_indices = onp.concatenate([[j for _ in range(self.k)] for j in range(self.N)])
                batch_y_entry[range(self.k * self.N), ground_truth_indices] = 1
                unstacked_y_batch.append(batch_y_entry)
        else:
            unstacked_y_batch = [onp.concatenate([[j for _ in range(self.k)] for j in range(self.N)]) for t in range(len(tasks))]
        
        y_batch = onp.stack(unstacked_y_batch)
        
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
        # print(predictions.shape, "ground", ground_truth.shape)

        if self.one_hot_ground_truth:
            loss = np.mean(-np.log(np.multiply(predictions, ground_truth)))
        else:
            losses = [-np.log(predictions[e][ground_truth[e]]) for e in range(len(predictions))]
            loss = sum(losses) / len(predictions)
        return loss

    def _get_accuracy(self, logits: np.ndarray, ground_truth:np.ndarray, return_plot=False):
        """
        Computes accuracy of a batch of predictions.

        :param logits: N x k batch of logits
        :param ground_truth: (N x k) ground truth indices
        """
        predictions = onp.array([int(np.argmax(i)) for i in logits]).reshape((self.N, self.k))
        accuracy_matrix = predictions == onp.array(ground_truth).reshape(self.N, self.k)
        accuracy = np.sum(accuracy_matrix) / (onp.prod(accuracy_matrix.shape))
        if return_plot:
            return accuracy, accuracy_matrix
        else:
            return accuracy

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

        final_predictions_array = self.network_forward(model_iterations[-1], validation_x)
        final_predictions_accuracy, final_accuracy_matrix = self._get_accuracy(logits=final_predictions_array, ground_truth=validation_y, return_plot=True)

        final_prediction_axis = fig.add_subplot(grid_spec[self.N, :])
        final_prediction_axis.imshow(final_accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        final_prediction_axis.set_title("Accuracy: {}".format(final_predictions_accuracy))

        if visualise_all:
            for i, (model_iteration) in enumerate(model_iterations[1:-1]):
                y_prediction_array = self.network_forward(model_iteration, validation_x)
                y_prediction_accuracy, accuracy_matrix = self._get_accuracy(logits=y_prediction_array, ground_truth=validation_y, return_plot=True)
                prediction_axis = fig.add_subplot(grid_spec[self.N + 1 + i, :])
                prediction_axis.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=1)

        plt.close()

        return fig

    def _get_fixed_validation_tasks(self):
        """
        If using fixed validation this method returns a set of tasks that are 
        equally spread across the task distribution space.
        """
        raise NotImplementedError

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
            first_approx = round(onp.sqrt(N))
            while N % first_approx > 0:
                first_approx -= 1
            return (int(first_approx), int(N / first_approx))

        if type(self._queue) == onp.ndarray:
            if len(self._queue.shape) <= 2:

                if feature == 'losses':
                    plot_queue = copy.deepcopy(self._queue)
                elif feature == 'counts':
                    plot_queue = copy.deepcopy(self.sample_counts)
                else:
                    raise ValueError("feature type not recognised. Use 'losses' or 'counts'")

                if len(self._queue.shape) == 1:
                    plot_queue = plot_queue.reshape((len(plot_queue), 1))

                # re-order queue to sorted by loss (makes visual more interpretable)
                plot_queue = plot_queue[onp.argsort(self._queue)]

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

        hist, bin_edges = onp.histogram(all_losses, bins=int(0.1 * len(all_losses)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = plt.figure()
        plt.plot(bin_centers, hist)
        return fig
