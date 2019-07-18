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
        self.params = params
        nn.Module.__init__(self)
        self.construct_layers()

    @abstractmethod
    def construct_layers(self):
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Base class abstract method")


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

        self.meta_optimiser = optim.Adam(self.model.parameters(), lr=self.meta_lr)

    @abstractmethod
    def _sample_task(self):
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _generate_batch(self, task, batch_size=25):
        """
        generates an array, x_batch, of B datapoints sampled randomly between domain_bounds
        and computes the sin of each point in x_batch to produce y_batch.
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _compute_loss(self, prediction, ground_truth):
        raise NotImplementedError("Base class abstract method")

    def outer_training_loop(self):
        meta_update_gradients = []
        for task in range(self.task_batch_size):
            meta_update_gradient = self.inner_training_loop()
            meta_update_gradients.append(meta_update_gradient)

        # meta update
        for meta_update_gradient in meta_update_gradients:
            model_state_dict = self.model.state_dict()
            for parameter_name, parameter in model_state_dict.items():
                # Transform the parameter using...
                transformed_parameter = parameter + self.meta_lr * meta_update_gradient[parameter_name]
                model_state_dict[parameter_name].copy_(transformed_parameter)

    def inner_training_loop(self):
        model_copy = copy.deepcopy(self.model)
        inner_optimiser = optim.Adam(model_copy.parameters(), lr=self.inner_update_lr)
        task = self._sample_task()
        x_batch, y_batch = self._generate_batch(task=task, batch_size=self.inner_update_batch_size)
        for _ in range(self.num_inner_updates):
            prediction = model_copy.forward(x_batch)
            loss = self._compute_loss(prediction, y_batch)
            inner_optimiser.zero_grad()
            loss.backward()
            inner_optimiser.step()

        meta_update_samples_x, meta_update_samples_y = self._generate_batch(task=task, batch_size=self.inner_update_batch_size)
        meta_update_prediction = model_copy.forward(meta_update_samples_x)
        meta_update_loss = self._compute_loss(meta_update_prediction, meta_update_samples_y)
        meta_update_loss.backward()
        meta_update_gradients = {param_name: param.grad for param_name, param in model_copy.named_parameters()}
        print("----inner_loss", meta_update_loss)
        print("----inner network copy params", list(model_copy.parameters()))
        return meta_update_gradients

    def train(self):
        for training_loop in range(self.training_iterations):
            if training_loop % self.validation_frequency == 0 and training_loop != 0:
                if self.checkpoint_path:
                    self.checkpoint_model()
                self.validate()
            # t0 = time.time()
            self.outer_training_loop()
            # print(time.time() - t0)

    def validate(self, visualise=True):
        overall_validation_loss = 0
        for _ in range(self.validation_task_batch_size):
            validation_network = copy.deepcopy(self.model)
            validation_optimiser = optim.Adam(validation_network.parameters(), lr=self.inner_update_lr)
            validation_task = self._sample_task()
            validation_x_batch, validation_y_batch = self._generate_batch(task=validation_task, batch_size=self.inner_update_batch_size)
            for _ in range(self.num_inner_updates):
                validation_prediction = validation_network.forward(validation_x_batch)
                validation_loss = self._compute_loss(validation_prediction, validation_y_batch)
                validation_optimiser.zero_grad()
                validation_loss.backward()
                validation_optimiser.step()
            test_task = self._sample_task()
            test_x_batch, test_y_batch = self._generate_batch(task=validation_task, batch_size=self.inner_update_batch_size)
            test_prediction = validation_network(test_x_batch)
            test_loss = self._compute_loss(test_prediction, test_y_batch)
            overall_validation_loss += float(test_loss)
        print('--- validation loss', overall_validation_loss / self.validation_task_batch_size)
        if visualise:
            self.visualise()

    def checkpoint_model(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
        PATH = '{}model_checkpoint_{}.pt'.format(self.checkpoint_path, timestamp)
        torch.save(self.model.state_dict(), PATH)

    @abstractmethod
    def visualise(self):
        """
        Allow for visualisation of test case. 
        E.g. a function plot for regression or a rollout for RL
        """
        raise NotImplementedError("Base class abstract method")