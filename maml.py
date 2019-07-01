import random
import copy

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn

class ModelNetwork(nn.Module):
    #TODO: Make this an abstract base class from which regression and classification (and RL) task specific MAML model calsses can be constructed
    def __init__(self, args):
        self.args = args
        nn.Module.__init__(self)
        self.construct_layers()

    @abstractmethod
    def construct_layers(self):
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Base class abstract method")


class MAML:
    #TODO: Make this an abstract base class? Or no need if the model class is abstract and this training orchestration class can be kept general.

    def __init__(self, args):
        self.args = args
        self.meta_optimiser = optim.Adam(self.model.parameters(), lr=self.args.meta_lr)

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
        meta_update_losses = []
        for task in range(self.args.task_batch_size):
            meta_update_loss = self.inner_training_loop()
            meta_update_losses.append(meta_update_loss)

        # meta update
        for meta_update_loss in meta_update_losses:
            self.meta_optimiser.zero_grad()
            meta_update_loss.backward()
            self.meta_optimiser.step()

    def inner_training_loop(self):
        model_copy = copy.deepcopy(self.model)
        inner_optimiser = optim.Adam(model_copy.parameters(), lr=self.args.inner_update_lr)
        task = self._sample_task()
        x_batch, y_batch = self._generate_batch(task=task, batch_size=self.args.inner_update_batch_size)
        for _ in range(self.args.num_inner_updates):
            prediction = model_copy.forward(x_batch)
            loss = self._compute_loss(prediction, y_batch)
            inner_optimiser.zero_grad()
            loss.backward()
            inner_optimiser.step()

        meta_update_samples_x, meta_update_samples_y = self._generate_batch(task=task, batch_size=self.args.inner_update_batch_size)
        meta_update_prediction = model_copy.forward(meta_update_samples_x)
        meta_update_loss = self._compute_loss(meta_update_prediction, meta_update_samples_y)
        return meta_update_loss

    def train(self):
        for training_loop in range(self.args.training_iterations):
            if training_loop % self.args.validation_frequency == 0:
                self.validate()
            t0 = time.time()
            self.outer_training_loop()
            print(time.time() - t0)

    def validate(self, plot=True):
        overall_validation_loss = 0
        for _ in range(self.args.validation_task_batch_size):
            validation_network = copy.deepcopy(self.model)
            validation_optimiser = optim.Adam(validation_network.parameters(), lr=self.args.inner_update_lr)
            validation_task = self._sample_task()
            validation_x_batch, validation_y_batch = self._generate_batch(task=validation_task, batch_size=self.args.inner_update_batch_size)
            for _ in range(self.args.num_inner_updates):
                validation_prediction = validation_network.forward(validation_x_batch)
                validation_loss = self._compute_loss(validation_prediction, validation_y_batch)
                validation_optimiser.zero_grad()
                validation_loss.backward()
                validation_optimiser.step()
            test_task = self._sample_task()
            test_x_batch, test_y_batch = self._generate_batch(task=validation_task, batch_size=self.args.inner_update_batch_size)
            test_prediction = validation_network(test_x_batch)
            test_loss = self._compute_loss(test_prediction, test_y_batch)
            overall_validation_loss += float(test_loss)
        print('--- validation loss', overall_validation_loss / self.args.validation_task_batch_size)
        if plot:
            self.plot_test()

    def plot_test(self, domain_bounds=(-5, 5)):
        """
        For regression tasks
        """
        test_network = copy.deepcopy(self.model)
        test_optimiser = optim.Adam(test_network.parameters(), lr=self.args.inner_update_lr)
        plot_test_task = self._sample_task()
        test_x_batch, test_y_batch = self._generate_batch(task=plot_test_task, batch_size=self.args.inner_update_batch_size)
        for _ in range(self.args.num_inner_updates):
            test_prediction = test_network.forward(test_x_batch)
            test_loss = self._compute_loss(test_prediction, test_y_batch)
            test_optimiser.zero_grad()
            test_loss.backward()
            test_optimiser.step()
        plot_x = np.linspace(domain_bounds[0], domain_bounds[1], 100)
        plot_y_ground_truth = [plot_test_task(xi) for xi in plot_x]
        plot_y_prediction = test_network(torch.tensor([[x] for x in plot_x]).to(self.device))
        # print(plot_y_prediction.detach().numpy())
        fig = plt.figure()
        plt.plot(plot_x, plot_y_prediction.cpu().detach().numpy(), linestyle='dashed')
        plt.plot(plot_x, plot_y_ground_truth)
        plt.scatter(test_x_batch.cpu(), test_y_batch.cpu(), marker='o')
        fig.savefig('prediction_test.png')
        plt.close()