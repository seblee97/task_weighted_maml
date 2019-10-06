import random
import copy
import time
import os
import datetime
import math
import warnings
import numpy as onp
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from typing import Any, Tuple, List, Dict

from abc import ABC, abstractmethod

from utils.priority import PriorityQueue

# jax imports
import jax.numpy as np
import jax
from jax import vmap # for auto-vectorizing functions
from jax import jit # for compiling functions for speedup
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
from jax import random
from jax.experimental import optimizers

from functools import partial # for use with vmap


class MAML(ABC):
    """
    Base class for the MAML algorithm.

    Includes training and validation loop methods.
    """
    def __init__(self, params):
        self.params = params

        # extract relevant parameters
        self.task_batch_size = self.params.get("task_batch_size")
        self.inner_update_lr = self.params.get("inner_update_lr")
        self.meta_lr = self.params.get("meta_lr")
        self.inner_update_k = self.params.get("inner_update_k")
        self.validation_k = self.params.get("validation_k")
        self.test_k = self.params.get("test_k")
        self.num_inner_updates = self.params.get("num_inner_updates")
        self.validation_num_inner_updates = self.params.get("validation_num_inner_updates")
        self.training_iterations = self.params.get("training_iterations")
        self.validation_frequency = self.params.get("validation_frequency")
        self.visualisation_frequency = self.params.get("visualisation_frequency")
        self.visualise_all = self.params.get("visualise_all")
        self.checkpoint_path = self.params.get("checkpoint_path")
        self.validation_task_batch_size = self.params.get("validation_task_batch_size")
        self.fixed_validation = self.params.get("fixed_validation")
        self.priority_sample = self.params.get("priority_sample")
        self.input_dimension = self.params.get("input_dimension")
        self.framework = self.params.get("framework")
        self.network_layers = self.params.get("network_layers")
        self.output_dimension = self.params.get("output_dimension")
        self.sample_type = self.params.get(["priority_queue", "sample_type"])

        # initialise tensorboard writer
        self.writer = SummaryWriter(self.checkpoint_path)
        # 'results/{}/{}'.format(self.params.get("experiment_name"), self.params.get("experiment_timestamp"))

        # if using priority queue for inner loop sampling, initialise 
        if self.params.get("priority_sample"):
            self.priority_queue = self._get_priority_queue()

        # write copy of config_yaml in model_checkpoint_folder
        self.params.save_configuration(self.checkpoint_path)

        # initialise jax model
        self.network_initialisation, self.network_forward = self._get_model()
        input_shape = [-1]
        for dim in self.input_dimension:
            input_shape.append(dim)
        input_shape = tuple(input_shape)
        random_initialisation = random.PRNGKey(0)

        self.network_input_shape = input_shape

        # load previously trained model to continue with
        if self.params.get(["resume", "model"]):
            model_checkpoint_path = self.params.get(["resume", "model"])
            try:
                print("Loading and resuming training from checkpoint @ {}".format(model_checkpoint_path))
                model_checkpoint = np.load(model_checkpoint_path, allow_pickle=True)[()]
                self.start_iteration = model_checkpoint["step"] #.split('_')[-1].split('.')[0])
                network_parameters = model_checkpoint["network_parameters"]
            except:
                raise FileNotFoundError("Resume checkpoint specified in config does not exist.")
        else:
            self.start_iteration = 0
            output_shape, network_parameters = self.network_initialisation(random_initialisation, input_shape)

        # initialise jax optimiser
        self.optimiser_initialisation, self.optimiser_update, self.get_params_from_optimiser = self._get_optimiser()
        self.optimiser_state = self.optimiser_initialisation(network_parameters)

    @abstractmethod
    def _get_model(self):
        """
        Return jax network initialisation and forward method.
        """
        raise NotImplementedError("Base class method")

    def _checkpoint_model(self, step_count: int, network_parameters: List) -> None:
        """
        Save a copy of the network parameters up to this point in training

        :param step_count: iteration number of training (meta-steps)
        :param network_parameters: parameters of network to save
        """
        os.makedirs(self.checkpoint_path, exist_ok=True)
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
        # format of model chekcpoint path: timestamp _ step_count
        PATH = '{}model_checkpoint_{}_{}.npy'.format(self.checkpoint_path, timestamp, str(step_count))
        np.save(PATH, {
            'step': step_count,
            'network_parameters': network_parameters
            })
        
    @abstractmethod
    def _get_optimiser(self):
        """
        Return jax optimiser: initialisation, update method and parameter getter method
        Optimiser learning rate is given by config (meta_lr).
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _get_priority_queue(self):
        """Initiate priority queue"""
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _sample_task(self, batch_size: int, validate:bool=False, step_count=None):
        """
        Sample specific task(s) from defined distribution of tasks 
        E.g. one specific sine function from family of sines

        :param batch_size: number of tasks to sample
        :param validate: whether or not tasks are being used for validation
        :param step_count: step count during training 

        :return tasks: batch of tasks
        :return task_indices: indices of priority queue associated with batch of tasks
        :return task_probabilities: probabilities of tasks sampled being chosen a priori

        Return type dependent of task family
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _get_task_from_params(self, parameters: List) -> Any:
        """
        Get specific task from specific given parameters 
        E.g. one specific sine function from family of sines

        :param parameters: parameters defining the specific task in the distribution

        Return type dependent of task family

        (method differs from _sample_task in that it is not a random sample but
        defined by parameters given)
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _generate_batch(self, tasks: List):
        """
        Obtain batch of training examples from a list of tasks

        :param tasks: list of tasks for which data points need to be sampled
        
        :return x_batch: x points sampled from data
        :return y_batch: y points associated with x_batch
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _compute_loss(self, parameters, inputs, ground_truth):
        """ 
        Compute loss for prediction based on ground truth

        :param parameters: current parameters of model
        :param inputs: x values on which to compute predictions and compute loss
        :param ground_truth: y value ground truth associated with inputs
        """
        raise NotImplementedError("Base class abstract method")

    def _inner_loop_update(self, parameters: List, x_batch: np.ndarray, y_batch: np.ndarray) -> List:
        """
        Inner loop of MAML algorithm, consists of optimisation steps on sampled tasks

        :param parameters: current parameters of model
        :param x_batch: batch of sampled data for each task
        :param y_batch: ground truth y points associated with x_batch

        :return updated_inner_parameters: updated inner network parameters
        """
        gradients = jax.grad(self._compute_loss)(parameters, x_batch, y_batch)
        inner_sgd_fn = lambda g, state: (state - self.inner_update_lr * g)
        updated_inner_parameters = jax.tree_util.tree_multimap(inner_sgd_fn, gradients, parameters)

        return updated_inner_parameters

    def _maml_loss(self, parameters: List, x_batch: np.ndarray, y_batch: np.ndarray, x_meta, y_meta, task_probability_weights: List):
        """
        Calculates loss to be backpropagated through meta network.

        :param parameters: current parameters of model
        :param x_batch: batch of sampled data for each task
        :param y_batch: ground truth y points associated with x_batch
        :param x_meta: batch of sampled data to be used for meta update (i.e. to compute loss after fine-tuning)
        :param y_meta: ground truth y points associated with x_meta
        :param task_probability_weights: importance weights to be used in importance sampling regime (None if not being used)
        """
        for _ in range(self.num_inner_updates):
            parameters = self._inner_loop_update(parameters, x_batch, y_batch)
        if task_probability_weights is not None:
            loss_for_meta_update = task_probability_weights * self._compute_loss(parameters, x_meta, y_meta)
        else:
            loss_for_meta_update = self._compute_loss(parameters, x_meta, y_meta)
        return loss_for_meta_update

    def batch_maml_loss(self, parameters, x_batch, y_batch, x_meta, y_meta, task_probability_weights, get_all_losses=False):
        """
        Batched version of _maml_loss method.

        :param get_all_losses: whether or not to return list of losses or mean over losses. If using priority queue, 
                               we require individual task losses.
        """
        task_losses = vmap(partial(self._maml_loss, parameters))(x_batch, y_batch, x_meta, y_meta, task_probability_weights)
        if get_all_losses:
            return task_losses
        return np.mean(task_losses)

    def outer_training_loop(self, step_count: int, optimiser_state, x_batch: np.array, y_batch: np.array, x_meta: np.array, y_meta: np.array, task_probability_weights: np.array):
        """
        Outer loop of MAML algorithm, consists of multiple inner loops and a meta update step

        :param step_count: iteration number
        :param optimiser_state: current state of optimiser
        :param x_batch: input data for batch on which to train inner loop
        :param y_batch: ground truth of batch label values 
        :param x_meta: extra input data sample for meta backprop
        :param y_meta: labels for extra input data
        :param task_probability_weights: weights for individual task losses

        :return updated_optimiser: new optimiser state
        :return parameters: parameters after outer loop step
        """
        # get parameters of current state of outer model
        parameters = self.get_params_from_optimiser(optimiser_state)

        # take derivative of inner loss term wrt outer model parameters (automatically wrt 'parameters' via jax.grad as 'parameters' is 1st arg of maml_loss)
        derivative_fn = jax.grad(self.batch_maml_loss)

        # evaluate derivative fn
        gradients = derivative_fn(parameters, x_batch, y_batch, x_meta, y_meta, task_probability_weights)

        # make step in outer model optimiser
        updated_optimiser = self.optimiser_update(step_count, gradients, optimiser_state)

        return updated_optimiser, parameters

    def fast_outer_training_loop(self):
        """
        jit accelerated outer loop method
        """
        return jit(self.outer_training_loop)

    def train(self):
        """
        Training orchestration method, calls outer loop and validation methods
        """
        print("Training starting...")
        for step_count in range(self.start_iteration, self.start_iteration + self.training_iterations):
            print("Training Step: {}".format(step_count))
            if step_count % self.validation_frequency == 0 and step_count != 0:
                if self.checkpoint_path:
                    current_network_parameters = self.get_params_from_optimiser(self.optimiser_state)
                    self._checkpoint_model(step_count=step_count, network_parameters=current_network_parameters)
                if self.priority_sample:
                    self.priority_queue.save_queue(step_count=step_count)
                if step_count % self.visualisation_frequency == 0:
                    vis = True
                else:
                    vis = False
                self.validate(step_count=step_count, visualise=vis)

            batch_of_tasks, max_indices, task_probabilities = self._sample_task(batch_size=self.task_batch_size, step_count=step_count)

            if 'importance' in self.sample_type:
                standard_task_probability = 1. / onp.prod(self.priority_queue.get_queue().shape)
                task_importance_weights = standard_task_probability / task_probabilities
                self.writer.add_scalar('queue_metrics/importance_weights_mean', float(onp.mean(task_importance_weights)), step_count)
            else:
                task_importance_weights = None

            x_train, y_train = self._generate_batch(batch_of_tasks)
            x_meta, y_meta = self._generate_batch(batch_of_tasks)

            self.optimiser_state, parameters = self.outer_training_loop(step_count, self.optimiser_state, x_train, y_train, x_meta, y_meta, task_probability_weights=task_importance_weights)
            
            # get a validation loss (mostly for logging purposes)
            meta_loss = onp.asarray(self.batch_maml_loss(parameters, x_train, y_train, x_meta, y_meta, task_probability_weights=None, get_all_losses=True))
            print("loss: ", float(np.mean(meta_loss)))

            if self.priority_sample:
                for t in range(len(meta_loss)):
                    self.priority_queue.insert(key=max_indices[t], data=meta_loss[t])

            self.writer.add_scalar('meta_metrics/meta_update_loss_mean', float(np.mean(meta_loss)), step_count)
            self.writer.add_scalar('meta_metrics/meta_update_loss_std', float(np.std(meta_loss)), step_count)

        net_params = self.get_params_from_optimiser(self.optimiser_state)

    def validate(self, step_count: int, visualise: bool=True) -> None:
        """
        Performs a validation step for loss during training. Also makes plots for tensorboard.

        :param step_count: number of steps in training undergone (used for print statement)
        :param visualise: whether or not to visualise validation run
        """
        validation_losses = []
        validation_figures = []
        validation_accuracies = []

        validation_parameter_tuples, validation_tasks = self._get_validation_tasks()

        for r, val_task in enumerate(validation_tasks):

            # initialise list of model iterations (used for visualisation of fine-tuning)
            validation_model_iterations = []

            # make copy of current state of outer model to fine tune for validation
            network_parameters = copy.deepcopy(self.get_params_from_optimiser(self.optimiser_state))

            # sample a task for validation fine-tuning
            validation_x_batch, validation_y_batch = self._generate_batch(tasks=[val_task])

            # sample a new batch from same validation task for testing fine-tuned model
            test_x_batch, test_y_batch = self._generate_batch(tasks=[val_task])
            
            if len(validation_x_batch.shape) != len(self.network_input_shape):
                # if there is an additional batch dimension for training, squeeze
                validation_x_batch = np.squeeze(validation_x_batch, axis=0)
                validation_y_batch = np.squeeze(validation_y_batch, axis=0)
                test_x_batch = np.squeeze(test_x_batch, axis=0)
                test_y_batch = np.squeeze(test_y_batch, axis=0)

            validation_model_iterations.append(copy.deepcopy(network_parameters))

            # inner loop update
            for _ in range(self.validation_num_inner_updates):

                network_parameters = self._inner_loop_update(network_parameters, validation_x_batch, validation_y_batch)
                
                validation_model_iterations.append(copy.deepcopy(network_parameters))

            test_loss = self._compute_loss(network_parameters, test_x_batch, test_y_batch)
            if self.is_classification:
                test_accuracy = self._get_accuracy(self.network_forward(network_parameters, validation_x_batch), validation_y_batch)
                validation_accuracies.append(test_accuracy)

            validation_losses.append(float(test_loss))

            if visualise:
                save_name = 'validation_step_{}_rep_{}.png'.format(step_count, r)
                validation_fig = self._visualise(
                    validation_model_iterations, val_task, validation_x_batch, validation_y_batch, save_name=save_name, visualise_all=self.visualise_all
                    )
                validation_figures.append(validation_fig)
                self.writer.add_figure("vadliation_plots/repeat_{}".format(r), validation_fig, step_count)

        mean_validation_loss = onp.mean(validation_losses)
        var_validation_loss = onp.std(validation_losses)
        mean_validation_accuracies = onp.mean(validation_accuracies)
        
        # get validation loss distribution
        validation_loss_distribution_fig = self._get_validation_loss_distribution_plot(validation_losses)
        # write validation loss distribution figure to tensorboard
        self.writer.add_figure("validation_loss_distribution", validation_loss_distribution_fig, step_count)

        print('--- validation loss @ step {}: {}'.format(step_count, mean_validation_loss))
        self.writer.add_scalar('meta_metrics/validation_loss_mean', mean_validation_loss, step_count)
        self.writer.add_scalar('meta_metrics/validation_loss_std', var_validation_loss, step_count)
        if self.is_classification:
            print('--- validation accuracy @ step {}: {}'.format(step_count, mean_validation_accuracies))
            self.writer.add_scalar('meta_metrics/validation_accuracy_mean', mean_validation_accuracies, step_count)

        # get validation loss heatmap as function of parameters governing validation task
        if self.fixed_validation and len(validation_parameter_tuples[0]) == 2:
            validation_loss_heatmap_fig = self._get_validation_loss_heatmap(validation_parameter_tuples, validation_losses)
            self.writer.add_figure("validation_loss_heatmap", validation_loss_heatmap_fig, step_count)
        else:
            warnings.warn("Visualisation of validation losses with parameter space dimension > 2 not supported", Warning)

        if self.priority_sample:
            # get figures from priority queue
            priority_queue_fig = self.priority_queue.visualise_priority_queue(feature='losses')
            priority_queue_count_fig = self.priority_queue.visualise_priority_queue(feature='counts')
            priority_queue_loss_dist_fig = self.priority_queue.visualise_priority_queue_loss_distribution()
            
            # write figures from priority queue to tensorboard
            if priority_queue_fig:
                self.writer.add_figure("priority_queue", priority_queue_fig, step_count)
            if priority_queue_count_fig:
                self.writer.add_figure("queue_counts", priority_queue_count_fig, step_count)
            if priority_queue_loss_dist_fig:
                self.writer.add_figure("queue_loss_dist", priority_queue_loss_dist_fig, step_count)

    @abstractmethod
    def _visualise(
        self, validation_model_iterations: List, val_task, validation_x_batch: np.ndarray, validation_y_batch: np.ndarray, 
        save_name: str, visualise_all: bool
        ):
        """
        Visualise qualitative run.

        :param validation_model_iterations: parameters of model after successive fine-tuning steps
        :param val_task: task being evaluated
        :param validation_x_batch: k data points fed to model for finetuning
        :param validation_y_batch: ground truth data associated with validation_x_batch
        :param save_name: name of file to be saved
        :param visualise_all: whether to visualise all fine-tuning steps or just final 
        """
        raise NotImplementedError("Base class method")

    def _get_validation_tasks(self):
        """produces set of tasks for use in validation"""
        if self.fixed_validation:
            return self._get_fixed_validation_tasks()
        else:
            return None, self._sample_task(batch_size=self.validation_task_batch_size, validate=True)[0]

    @abstractmethod
    def _get_fixed_validation_tasks(self):
        """
        If using fixed validation this method returns a set of tasks that are 
        equally spread across the task distribution space.
        """
        raise NotImplementedError("Base class method")

    def _get_validation_loss_distribution_plot(self, validation_losses):
        """returns matplotlib figure showing distribution of validation_losses"""
        hist, bin_edges = onp.histogram(validation_losses, bins=int(0.1 * len(validation_losses)))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = plt.figure()
        plt.plot(bin_centers, hist)

        return fig

    def _get_validation_loss_heatmap(self, validation_parameter_tuples, validation_losses):
        """returns matplotlib figure showing heatmap of vadliation losses as function of parameter space"""
        unique_parameter_range_lens = []

        for i in range(2):
            unique_parameter_range_lens.append(len(onp.unique([p[i] for p in validation_parameter_tuples])))
        validation_losses_grid = onp.array(validation_losses).reshape(tuple(unique_parameter_range_lens))

        fig = plt.figure()
        plt.imshow(validation_losses_grid)
        plt.colorbar()

        return fig
 