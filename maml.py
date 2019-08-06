import random
import copy
import time
import os
import datetime

from tensorboardX import SummaryWriter

from typing import Any, Tuple, List, Dict

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn

from utils.priority import PriorityQueue

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

        # if using priority queue for inner loop sampling, initialise 
        if self.params.get("priority_sample"):
            self.priority_queue = PriorityQueue(
                sample_type=self.params.get(["priority_queue", "sample_type"]),
                block_sizes=self.params.get(["priority_queue", "block_sizes"]),
                burn_in=self.params.get(["priority_queue", "burn_in"])
                )

            if self.params.get(["priority_queue", "burn_in"]) is not None:
                raise NotImplementedError("Burn in functionality not yet implemented")
                self._fill_pq_buffer()

    @abstractmethod
    def _construct_layers(self) -> None:
        """
        Build up layers (weights and biases) of network using dimensions specified in configuration
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass of x through network

        :param x: tensor to be fed through network
        """
        raise NotImplementedError("Base class abstract method")

    def _reset_parameters(self) -> None:
        """
        Reset all parameters in network using unifrom Gaussian initialisation
        """
        for l in range(len(self.layer_dimensions) - 1):
            std = 1. / np.sqrt(self.layer_dimensions[l])
            self.weights[l].data.uniform_(-std, std) # uniform Gaussian initialisation
            self.biases[l].data.uniform_(-std, std)


class MAML(ABC):

    def __init__(self, params):
        self.params = params

        self.writer = SummaryWriter() 

        # extract relevant parameters
        self.task_batch_size = self.params.get("task_batch_size")
        self.inner_update_lr = self.params.get("inner_update_lr")
        self.meta_lr = self.params.get("meta_lr")
        self.inner_update_batch_size = self.params.get("inner_update_batch_size")
        self.num_inner_updates = self.params.get("num_inner_updates")
        self.validation_num_inner_updates = self.params.get("validation_num_inner_updates")
        self.training_iterations = self.params.get("training_iterations")
        self.validation_frequency = self.params.get("validation_frequency")
        self.checkpoint_path = self.params.get("checkpoint_path")
        self.validation_task_batch_size = self.params.get("validation_task_batch_size")
        self.fixed_validation = self.params.get("fixed_validation")

        # load previously trained model to continue with
        if self.params.get("resume"):
            model_checkpoint = self.params.get("resume")
            try:
                print("Loading and resuming training from checkpoint @ {}".format(model_checkpoint))
                self.model_inner.load_state_dict(torch.load(model_checkpoint))
                self.start_iteration = float(model_checkpoint.split('_')[-1])
            except:
                raise FileNotFoundError("Resume checkpoint specified in config does not exist.")
        else:
            self.start_iteration = 0
        
        self.model_outer = copy.deepcopy(self.model_inner).to(self.device)

        self.meta_optimiser = optim.Adam(
            self.model_outer.weights + self.model_outer.biases, lr=self.meta_lr
            )

        # write copy of config_yaml in model_checkpoint_folder
        self.params.save_configuration(self.checkpoint_path)

    @abstractmethod
    def _sample_task(self) -> Any:
        """
        Sample specific task from defined distribution of tasks 
        E.g. one specific sine function from family of sines

        Return type dependent of task family
        """
        raise NotImplementedError("Base class abstract method")

    @abstractmethod
    def _get_task_from_params(self, parameters: Dict[str, Any]) -> Any:
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

    def outer_training_loop(self) -> None:
        """
        Outer loop of MAML algorithm, consists of multiple inner loops and a meta update step
        """

        # get copies of meta network parameters
        weight_copies = [w.clone() for w in self.model_outer.weights]
        bias_copies = [b.clone() for b in self.model_outer.biases]

        # initialise cumulative gradient to be used in meta update step
        meta_update_gradient = [0 for _ in range(len(weight_copies) + len(bias_copies))]

        for _ in range(self.task_batch_size):
            task_meta_gradient = self.inner_training_loop(weight_copies, bias_copies)  
            for i in range(len(weight_copies) + len(bias_copies)):
                meta_update_gradient[i] += task_meta_gradient[i].detach()

        # meta update
        # zero previously collected gradients
        self.meta_optimiser.zero_grad()

        for i in range(len(self.model_outer.weights)):
            self.model_outer.weights[i].grad = meta_update_gradient[i] / self.task_batch_size
            meta_update_gradient[i] = 0
        for j in range(len(self.model_outer.biases)):
            self.model_outer.biases[j].grad = meta_update_gradient[i + j + 1] / self.task_batch_size
            meta_update_gradient[i + j + 1] = 0

        self.meta_optimiser.step()

    def inner_training_loop(self, weight_copies: List[torch.Tensor], bias_copies: List[torch.Tensor]) -> torch.Tensor:
        """
        Inner loop of MAML algorithm, consists of optimisation steps on sampled tasks

        :param weight_copies: copy of weights in network of outer loop
        :param bias_copies: copy of biases in network of outer loop

        :return meta_update_grad: gradient to be fed to meta update
        """

        # reset network weights to meta network weights
        self.model_inner.weights = [w.clone() for w in weight_copies]
        self.model_inner.biases = [b.clone() for b in bias_copies]

        # sample a task from task distribution and generate x, y tensors for that task
        if self.priority_sample:
            task_parameters = self.priority_queue.query()
            task = self._get_task_from_params()
        else:
            task = self._sample_task()
        x_batch, y_batch = self._generate_batch(task=task, batch_size=self.inner_update_batch_size)

        for _ in range(self.num_inner_updates):

            # forward pass
            prediction = self.model_inner(x_batch)

            # compute loss
            loss = self._compute_loss(prediction, y_batch)

            # compute gradients wrt inner model copy
            gradients = torch.autograd.grad(loss, self.model_inner.weights + self.model_inner.biases, create_graph=True, retain_graph=True)

            # update inner model using current model 
            for i in range(len(self.model_inner.weights)):
                self.model_inner.weights[i] = self.model_inner.weights[i] - self.inner_update_lr * gradients[i].detach()
            for j in range(len(self.model_inner.biases)):
                self.model_inner.biases[j] = self.model_inner.biases[j] - self.inner_update_lr * gradients[i + j + 1].detach()

        # generate x, y tensors for meta update task sample
        meta_update_samples_x, meta_update_samples_y = self._generate_batch(task=task, batch_size=self.inner_update_batch_size)

        # forward pass for meta update
        meta_update_prediction = self.model_inner(meta_update_samples_x)

        # compute loss
        meta_update_loss = self._compute_loss(meta_update_prediction, meta_update_samples_y)

        if self.priority_sample:
            self.priority_queue.insert(key=task_parameters, data=meta_update_loss)

        # compute gradients wrt outer model (meta network)
        meta_update_grad = torch.autograd.grad(meta_update_loss, self.model_outer.weights + self.model_outer.biases)

        return meta_update_grad

    def train(self) -> None:
        """
        Training orchestration method, calls outer loop and validation methods
        """
        for step_count in range(self.start_iteration, self.start_iteration + self.training_iterations):
            if step_count % self.validation_frequency == 0 and step_count != 0:
                if self.checkpoint_path:
                    self.checkpoint_model(step_count=step_count)
                self.validate(step_count=step_count)
            # t0 = time.time()
            self.outer_training_loop()
            # print(time.time() - t0)

    def validate(self, step_count: int, visualise: bool=True) -> None:
        """
        Performs a validation step for loss during training

        :param step_count: number of steps in training undergone (used for pring statement)
        :param visualise: whether or not to visualise validation run
        """

        overall_validation_loss = 0
        validation_figures = []

        validation_tasks = self._get_validation_tasks()

        for r, val_task in enumerate(validation_tasks):

            # initialise list of model iterations (used for visualisation of fine-tuning)
            validation_model_iterations = []

            # make copies of outer network for use in validation
            validation_network = copy.deepcopy(self.model_outer).to(self.device)
            validation_optimiser = optim.Adam(validation_network.weights + validation_network.biases, lr=self.inner_update_lr)

            # sample a task for validation fine-tuning
            validation_x_batch, validation_y_batch = self._generate_batch(task=val_task, batch_size=self.inner_update_batch_size)

            validation_model_iterations.append(([w for w in validation_network.weights], [b for b in validation_network.biases]))

            # inner loop update
            for _ in range(self.validation_num_inner_updates):

                # prediction of validation batch
                validation_prediction = validation_network(validation_x_batch)

                # compute loss
                validation_loss = self._compute_loss(validation_prediction, validation_y_batch)

                # find gradients of validation loss wrt inner model weights
                validation_update_grad = torch.autograd.grad(validation_loss, validation_network.weights + validation_network.biases)

                # update inner model weights
                for i in range(len(validation_network.weights)):
                    validation_network.weights[i] = validation_network.weights[i] - self.inner_update_lr * validation_update_grad[i].detach()
                for j in range(len(validation_network.biases)):
                    validation_network.biases[j] = validation_network.biases[j] - self.inner_update_lr * validation_update_grad[i + j + 1].detach()

                current_weights = [w for w in validation_network.weights]
                current_biases = [w for w in validation_network.biases]
                validation_model_iterations.append((current_weights, current_biases))
            
            # sample a new batch from same validation task for testing fine-tuned model
            test_x_batch, test_y_batch = self._generate_batch(task=val_task, batch_size=self.inner_update_batch_size)

            test_prediction = validation_network(test_x_batch)
            test_loss = self._compute_loss(test_prediction, test_y_batch)

            overall_validation_loss += float(test_loss)

            if visualise:
                save_name = 'validation_step_{}_rep_{}.png'.format(step_count, r)
                validation_fig = self.visualise(
                    validation_model_iterations, val_task, validation_x_batch, validation_y_batch, save_name=save_name
                    )
                validation_figures.append(validation_fig)

        print('--- validation loss @ step {}: {}'.format(step_count, overall_validation_loss / self.validation_task_batch_size))
        self.writer.add_scalar('experiment/meta_validation_loss', overall_validation_loss / self.validation_task_batch_size, step_count)
        for f, fig in enumerate(validation_figures):
            self.writer.add_figure("vadliation_plots/repeat_{}".format(f), fig, step_count)

    def _get_validation_tasks(self):
        """produces set of tasks for use in validation"""
        if self.fixed_validation:
            return self._get_fixed_validation_tasks()
        else:
            return [self._sample_task() for _ in range(self.validation_task_batch_size)]

    @abstractmethod
    def _get_fixed_validation_tasks(self):
        """
        If using fixed validation this method returns a set of tasks that are 
        'representative' of the task distribution in some meaningful way.
        """
        raise NotImplementedError("Base class method")

    def checkpoint_model(self, step_count: int) -> None:
        """
        Save a copy of the outer model up to this point in training

        :param step_count: iteration number of training (meta-steps)
        """
        os.makedirs(self.checkpoint_path, exist_ok=True)
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
        # format of model chekcpoint path: timestamp _ step_count
        PATH = '{}model_checkpoint_{}_{}.pt'.format(self.checkpoint_path, timestamp, str(step_count))
        torch.save(self.model_outer.state_dict(), PATH)

    @abstractmethod
    def visualise(self) -> None:
        """
        Allow for visualisation of test case. 
        E.g. a function plot for regression or a rollout for RL
        """
        raise NotImplementedError("Base class abstract method")

        