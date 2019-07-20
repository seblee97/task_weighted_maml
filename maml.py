import random
import copy
import time
import os
import datetime

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn

class ModelNetwork(nn.Module):
    
    def __init__(self, params):
        nn.Module.__init__(self)

        self.params = params
        self.device = self.params.get("device")

        self.weights = []
        self.biases = []

        self.layer_dimensions = [self.params.get("input_dimension")] \
                                + self.params.get("network_layers") \
                                + [self.params.get("output_dimension")]

        self._construct_layers()

    @abstractmethod
    def _construct_layers(self):
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Base class abstract method")

    def _reset_parameters(self):
        for l in range(len(self.layer_dimensions) - 1):
            std = 1. / np.sqrt(self.layer_dimensions[l])
            self.weights[l].data.uniform_(-std, std) # uniform Gaussian initialisation
            self.biases[l].data.uniform_(-std, std)


class MAML(ABC):

    def __init__(self, params):
        self.params = params 

        # extract relevant parameters
        self.task_batch_size = self.params.get("task_batch_size")
        self.inner_update_lr = self.params.get("inner_update_lr")
        self.meta_lr = self.params.get("meta_lr")
        self.inner_update_batch_size = self.params.get("inner_update_batch_size")
        self.num_inner_updates = self.params.get("num_inner_updates")
        self.training_iterations = self.params.get("training_iterations")
        self.validation_frequency = self.params.get("validation_frequency")
        self.checkpoint_path = self.params.get("checkpoint_path")
        self.validation_task_batch_size = self.params.get("validation_task_batch_size")
        
        self.model_outer = copy.deepcopy(self.model_inner).to(self.device)

        self.meta_optimiser = optim.Adam(
            self.model_outer.weights + self.model_outer.biases, lr=self.meta_lr
            )

        # write copy of config_yaml in model_checkpoint_folder
        self.params.save_configuration(self.checkpoint_path)

    @abstractmethod
    def _sample_task(self):
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _generate_batch(self, task, batch_size=25):
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _compute_loss(self, prediction, ground_truth):
        raise NotImplementedError("Base class abstract method")

    def outer_training_loop(self):

        # get copies of meta network parameters
        weight_copies = [w.clone() for w in self.model_outer.weights]
        bias_copies = [b.clone() for b in self.model_outer.biases]

        # initialise cumulative gradient to be used in meta update step
        meta_update_gradients = []

        for _ in range(self.task_batch_size):
            meta_update_gradient = self.inner_training_loop(weight_copies, bias_copies)    
            meta_update_gradients.append(meta_update_gradient)

        # meta update
        for meta_update_gradient in meta_update_gradients:
            
            # zero previously collected gradients
            self.meta_optimiser.zero_grad()
            outer_model_state_dict = self.model_outer.state_dict()
            for parameter_name, parameter in outer_model_state_dict.items():
                transformed_parameter = parameter - self.meta_lr * meta_update_gradient[parameter_name] / self.task_batch_size # TODO: change variable name to tasks_per_meta_update
                outer_model_state_dict[parameter_name].copy_(transformed_parameter)

    def inner_training_loop(self, weight_copies, bias_copies):

        # reset network weights to meta network weights
        self.model_inner.weights = [w.clone() for w in weight_copies]
        self.model_inner.biases = [b.clone() for b in bias_copies]

        # sample a task from task distribution and generate x, y tensors for that task
        task = self._sample_task()
        x_batch, y_batch = self._generate_batch(task=task, batch_size=self.inner_update_batch_size)

        for _ in range(self.num_inner_updates):

            # forward pass
            prediction = self.model_inner(x_batch)

            # compute loss
            loss = self._compute_loss(prediction, y_batch)

            # compute gradients wrt inner model copy
            gradients = torch.autograd.grad(loss, self.model_inner.weights + self.model_inner.biases, create_graph=True, retain_graph=True)

            # update inner model using current model # TODO: HAVE YOU BUILT UP COMPUTATION GRAPH?
            for i in range(len(self.model_inner.weights)):
                self.model_inner.weights[i] = self.model_inner.weights[i] - self.inner_update_lr * gradients[i]
            for j in range(len(self.model_inner.biases)):
                self.model_inner.biases[j] = self.model_inner.biases[j] - self.inner_update_lr * gradients[i + j + 1] 

        # generate x, y tensors for meta update task sample
        meta_update_samples_x, meta_update_samples_y = self._generate_batch(task=task, batch_size=self.inner_update_batch_size)

        # forward pass for meta update
        meta_update_prediction = self.model_inner(meta_update_samples_x)

        # compute loss
        meta_update_loss = self._compute_loss(meta_update_prediction, meta_update_samples_y)

        # compute gradients wrt outer model (meta network)
        meta_update_grad = torch.autograd.grad(meta_update_loss, self.model_outer.weights + self.model_outer.biases)

        return meta_update_grad

    def train(self):
        for training_loop in range(self.training_iterations):
            if training_loop % self.validation_frequency == 0 and training_loop != 0:
                if self.checkpoint_path:
                    self.checkpoint_model()
                self.validate()
            # t0 = time.time()
            self.outer_training_loop()
            # print(time.time() - t0)

    def validate(self, visualise=False):
        overall_validation_loss = 0
        for _ in range(self.validation_task_batch_size):
            validation_network = copy.deepcopy(self.model_outer)
            validation_optimiser = optim.Adam(validation_network.weights + validation_network.biases, lr=self.inner_update_lr)
            validation_task = self._sample_task()
            validation_x_batch, validation_y_batch = self._generate_batch(task=validation_task, batch_size=self.inner_update_batch_size)
            for _ in range(self.num_inner_updates):
                validation_prediction = validation_network(validation_x_batch)
                validation_loss = self._compute_loss(validation_prediction, validation_y_batch)
                validation_update_grad = torch.autograd.grad(validation_loss, validation_network.weights + validation_network.biases)
                for i in range(len(validation_network.weights)):
                    validation_network.weights[i] = validation_network.weights[i] - self.inner_update_lr * validation_update_grad[i]
                for j in range(len(validation_network.biases)):
                    validation_network.biases[j] = validation_network.biases[j] - self.inner_update_lr * validation_update_grad[i + j + 1]
            final_validation_prediction = validation_network(validation_x_batch)
            final_validation_loss = self._compute_loss(final_validation_prediction, validation_y_batch)
            # test_task = self._sample_task()
            # test_x_batch, test_y_batch = self._generate_batch(task=validation_task, batch_size=self.inner_update_batch_size)
            # test_prediction = validation_network(test_x_batch)
            # test_loss = self._compute_loss(test_prediction, test_y_batch)
            overall_validation_loss += float(final_validation_loss)
        print('--- validation loss', overall_validation_loss / self.validation_task_batch_size)
        if visualise:
            self.visualise()

    def checkpoint_model(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
        PATH = '{}model_checkpoint_{}.pt'.format(self.checkpoint_path, timestamp)
        torch.save(self.model_outer.state_dict(), PATH)

    @abstractmethod
    def visualise(self):
        """
        Allow for visualisation of test case. 
        E.g. a function plot for regression or a rollout for RL
        """
        raise NotImplementedError("Base class abstract method")