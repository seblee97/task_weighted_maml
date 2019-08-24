import random
import copy
import time
import os
import datetime

from tensorboardX import SummaryWriter

from typing import Any, Tuple, List, Dict

from abc import ABC, abstractmethod

import numpy as onp
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn

from utils.priority import PriorityQueue

import jax.numpy as np
import numpy as np
import matplotlib.pyplot as plt
import jax

# jax imports
from jax import vmap # for auto-vectorizing functions
from jax import jit # for compiling functions for speedup
from jax.experimental import stax # neural network library
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax
from jax import random
from jax.experimental import optimizers

from functools import partial # for use with vmap


class MAML(ABC):

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

        # initialise tensorboard writer
        self.writer = SummaryWriter(self.checkpoint_path)
        # 'results/{}/{}'.format(self.params.get("experiment_name"), self.params.get("experiment_timestamp"))

        # if using priority queue for inner loop sampling, initialise 
        if self.params.get("priority_sample"):
            self.priority_queue = self._get_priority_queue()

        # write copy of config_yaml in model_checkpoint_folder
        self.params.save_configuration(self.checkpoint_path)

        self.network_initialisation, self.network_forward = self._get_model()
        input_shape = (-1, self.input_dimension,)
        random_initialisation = random.PRNGKey(0)
        output_shape, network_parameters = self.network_initialisation(random_initialisation, input_shape)

        self.optimier_initialisation, self.optimiser_update, self.get_params_from_optimiser = self._get_optimiser()
        self.optimiser_state = self.optimier_initialisation(network_parameters)

    @abstractmethod
    def _get_model(self):
        """
        Return jax network
        """
        raise NotImplementedError("Base class method")
        
    @abstractmethod
    def _get_optimiser(self):
        """
        Return jax optimiser
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _get_priority_queue(self):
        """Initiate priority queue"""
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _sample_task(self) -> Any:
        """
        Sample specific task from defined distribution of tasks 
        E.g. one specific sine function from family of sines

        Return type dependent of task family
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _get_task_from_params(self) -> Any:
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
    def _generate_batch(self, task: Any, batch_size: int=25) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain batch of training examples from a sampled task

        :param task: specific task from which to sample x, y pairs
        :param batch_size: number of x, y pairs to sample for batch
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _compute_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """ 
        Compute loss for prediction based on ground truth

        :param prediction: output of network for x
        :param ground_trugh: y value ground truth associated with x
        """
        raise NotImplementedError("Base class abstract method")

    def inner_loop_update(self, parameters, x, y):
        """
        Inner loop of MAML algorithm, consists of optimisation steps on sampled tasks

        :return
        """
        gradients = jax.grad(self._compute_loss)(parameters, x, y)
        inner_sgd_fn = lambda g, state: (state - self.inner_update_lr * g)
        return jax.tree_util.tree_multimap(inner_sgd_fn, gradients, parameters) # TODO (and docstring)

    def _maml_loss(self, parameters, x_batch, y_batch, x_validation, y_validation):
        for _ in range(self.num_inner_updates):
            parameters = self.inner_loop_update(parameters, x_batch, y_batch)
        loss_for_meta_update = self._compute_loss(parameters, x_validation, y_validation)
        return loss_for_meta_update

    def batch_maml_loss(self, parameters, x_batch, y_batch, x_validation, y_validation):
        task_losses = vmap(partial(self._maml_loss, parameters))(x_batch, y_batch, x_validation, y_validation)
        return np.mean(task_losses)

    def _step(self, i, optimiser_state, x_batch, y_batch, validation_x, validation_y):
        """
        """
        # import pdb; pdb.set_trace()
        # get parameters of current state of outer model
        parameters = self.get_params_from_optimiser(optimiser_state)

        # take derivative of inner loss term wrt outer model parameters (automatically wrt 'parameters' via jax.grad as 'parameters' is 1st arg of maml_loss)
        derivative_fn = jax.grad(self.batch_maml_loss)

        # evaluate derivative fn
        gradients = derivative_fn(parameters, x_batch, y_batch, validation_x, validation_y)

        # get a validation loss (mostly for logging purposes)
        validation_loss = self.batch_maml_loss(parameters, x_batch, y_batch, validation_x, validation_y)

        # make step in outer model optimiser
        updated_optimiser = self.optimiser_update(i, gradients, optimiser_state)

        return updated_optimiser, validation_loss

    def _fast_step(self):
        return jit(self._step)

    def train(self):

        validation_losses = []

        for i in range(self.training_iterations):
            batch_of_tasks = self._sample_task()
            x_train, y_train = self._generate_batch(batch_of_tasks)
            x_validation, y_validation = self._generate_batch(batch_of_tasks)
            self.optimiser_state, validation_loss = self._fast_step()(i, self.optimiser_state, x_train, y_train, x_validation, y_validation)
            validation_losses.append(validation_loss)
            if i % 1000 == 0:
                print(i, x_train.shape, validation_loss)
        net_params = self.get_params_from_optimiser(self.optimiser_state)
        