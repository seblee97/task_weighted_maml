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

        self._weights = []
        self._biases = []

        self.layer_dimensions = [self.params.get("input_dimension")] \
                                + self.params.get("network_layers") \
                                + [self.params.get("output_dimension")]

        self._construct_layers()

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
        for l in range(len(self._weights)):
            # if type(self.layer_dimensions[l]) == int:
            #     std = 1. / np.sqrt(self.layer_dimensions[l])
            # elif type(self.layer_dimensions[l]) == list:
            #     std = 1. / np.sqrt(np.prod(self.layer_dimensions[l]))
            # elif type(self.layer_dimensions[l]) == np.ndarray:
            #     std = 1. / np.sqrt(np.prod(self.layer_dimensions[l]))
            std = 1. / np.sqrt(np.prod(self._weights[l].shape))
            self._weights[l].data.uniform_(-std, std) # uniform Gaussian initialisation
            self._biases[l].data.uniform_(-std, std)


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

            if self.params.get(["priority_queue", "burn_in"]) is not None:
                raise NotImplementedError("Burn in functionality not yet implemented")
                self._fill_pq_buffer()

        # load previously trained model to continue with
        if self.params.get(["resume", "model"]):
            model_checkpoint = self.params.get(["resume", "model"])
            try:
                print("Loading and resuming training from checkpoint @ {}".format(model_checkpoint))
                checkpoint = torch.load(model_checkpoint)
                self.model_inner.load_state_dict(checkpoint['model_state_dict'])
                self.start_iteration = checkpoint['step'] # int(model_checkpoint.split('_')[-1].split('.')[0])
            except:
                raise FileNotFoundError("Resume checkpoint specified in config does not exist.")
        else:
            self.start_iteration = 0
        
        self.model_outer = copy.deepcopy(self.model_inner).to(self.device)

        self.meta_optimiser = optim.Adam(
            self.model_outer._weights + self.model_outer._biases, lr=self.meta_lr
            )

        if self.params.get(["resume", "model"]):
            self.meta_optimiser.load_state_dict(checkpoint['optimizer_state_dict'])

        # write copy of config_yaml in model_checkpoint_folder
        self.params.save_configuration(self.checkpoint_path)

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

    def _get_accuracy(self, logits: np.ndarray, ground_truth:np.ndarray, return_plot=False):
        """
        Computes accuracy of a batch of predictions.

        :param logits: N x k batch of logits
        :param ground_truth: (N x k) ground truth indices
        """
        np_logits = logits.detach().numpy()
        np_ground = ground_truth.detach().numpy()
        predictions = np.array([int(np.argmax(i)) for i in np_logits]).reshape((self.N, self.k))
        accuracy_matrix = predictions == np.array(np_ground).reshape(self.N, self.k)
        accuracy = np.sum(accuracy_matrix) / (np.prod(accuracy_matrix.shape))
        if return_plot:
            return accuracy, accuracy_matrix
        else:
            return accuracy

    def outer_training_loop(self, step_count: int) -> None:
        """
        Outer loop of MAML algorithm, consists of multiple inner loops and a meta update step
        """
        # get copies of meta network parameters
        weight_copies = [w.clone() for w in self.model_outer._weights]
        bias_copies = [b.clone() for b in self.model_outer._biases]

        # initialise cumulative gradient to be used in meta update step
        meta_update_gradient = [0 for _ in range(len(weight_copies) + len(bias_copies))]

        meta_loss = []
        task_importance_weights = []

        for _ in range(self.task_batch_size):
            task_meta_gradient, task_loss, importance_weight = self.inner_training_loop(step_count, weight_copies, bias_copies)
            meta_loss.append(task_loss) 
            task_importance_weights.append(importance_weight)
            for i in range(len(weight_copies) + len(bias_copies)):
                meta_update_gradient[i] += importance_weight * task_meta_gradient[i].detach()

        print("loss: ", float(np.mean(meta_loss)))

        self.writer.add_scalar('meta_metrics/meta_update_loss_mean', np.mean(meta_loss), step_count)
        self.writer.add_scalar('meta_metrics/meta_update_loss_std', np.std(meta_loss), step_count)
        self.writer.add_scalar('queue_metrics/importance_weights_mean', np.mean(task_importance_weights), step_count)

        # meta update
        # zero previously collected gradients
        self.meta_optimiser.zero_grad()

        for i in range(len(self.model_outer._weights)):
            self.model_outer._weights[i].grad = meta_update_gradient[i] / self.task_batch_size
            meta_update_gradient[i] = 0
        for j in range(len(self.model_outer._biases)):
            self.model_outer._biases[j].grad = meta_update_gradient[i + j + 1] / self.task_batch_size
            meta_update_gradient[i + j + 1] = 0

        self.meta_optimiser.step()

    def inner_training_loop(self, step_count: int, weight_copies: List[torch.Tensor], bias_copies: List[torch.Tensor]) -> torch.Tensor:
        """
        Inner loop of MAML algorithm, consists of optimisation steps on sampled tasks

        :param weight_copies: copy of weights in network of outer loop
        :param bias_copies: copy of biases in network of outer loop

        :return meta_update_grad: gradient to be fed to meta update
        """
        # reset network weights to meta network weights
        self.model_inner._weights = [w.clone() for w in weight_copies]
        self.model_inner._biases =  [b.clone() for b in bias_copies]

        # sample a task from task distribution and generate x, y tensors for that task
        if self.priority_sample:
            # query queue for next task parameters
            max_indices, task_parameters, task_probability = self.priority_queue.query(step=step_count)

            # get epsilon value
            epsilon = self.priority_queue.get_epsilon()

            # get task from parameters returned from query
            task = self._get_task_from_params(task_parameters)

            # compute metrics for tb logging
            queue_count_loss_correlation = self.priority_queue.compute_count_loss_correlation()
            queue_mean = np.mean(self.priority_queue.get_queue())
            queue_std = np.std(self.priority_queue.get_queue())

            # write to tensorboard
            if epsilon:
                self.writer.add_scalar('queue_metrics/epsilon', epsilon, step_count)
            self.writer.add_scalar('queue_metrics/queue_correlation', queue_count_loss_correlation, step_count)
            self.writer.add_scalar('queue_metrics/queue_mean', queue_mean, step_count)
            self.writer.add_scalar('queue_metrics/queue_std', queue_std, step_count)
        else:
            task = self._sample_task()
        x_batch, y_batch = self._generate_batch(task=task, batch_size=self.inner_update_k)

        for _ in range(self.num_inner_updates):

            # forward pass
            prediction = self.model_inner(x_batch)

            # compute loss
            loss = self._compute_loss(prediction, y_batch)

            # compute gradients wrt inner model copy
            inner_trainable_parameters = [w for w in self.model_inner._weights] + [b for b in self.model_inner._biases]
            gradients = torch.autograd.grad(loss, inner_trainable_parameters, create_graph=True, retain_graph=True)

            # update inner model using current model 
            for i in range(len(self.model_inner._weights)):
                self.model_inner._weights[i] = self.model_inner._weights[i] - self.inner_update_lr * gradients[i]
            for j in range(len(self.model_inner._biases)):
                self.model_inner._biases[j] = self.model_inner._biases[j] - self.inner_update_lr * gradients[i + j+ 1]

        # generate x, y tensors for meta update task sample
        meta_update_samples_x, meta_update_samples_y = self._generate_batch(task=task, batch_size=self.inner_update_k)

        # forward pass for meta update
        meta_update_prediction = self.model_inner(meta_update_samples_x)

        # compute loss
        meta_update_loss = self._compute_loss(meta_update_prediction, meta_update_samples_y)

        if self.priority_sample:
            # print("----------- max_indices", max_indices)
            # print("----------- task_parameters", task_parameters)
            # print("----------- meta_update_loss", meta_update_loss)
            self.priority_queue.insert(key=max_indices, data=meta_update_loss)

        # compute gradients wrt outer model (meta network)
        meta_update_grad = torch.autograd.grad(meta_update_loss, self.model_outer._weights + self.model_outer._biases)

        individual_loss = float(meta_update_loss)

        if 'importance' in self.sample_type:
            standard_task_probability = 1. / np.prod(self.priority_queue.get_queue().shape)
            importance_weight = standard_task_probability / task_probability
        else:
            importance_weight = 1.

        return meta_update_grad, individual_loss, importance_weight


    def train(self) -> None:
        """
        Training orchestration method, calls outer loop and validation methods
        """
        print("Training starting...")
        for step_count in range(self.start_iteration, self.start_iteration + self.training_iterations):
            print("Training Step: {}".format(step_count))
            t0 = time.time()
            if step_count % self.validation_frequency == 0 and step_count != 0:
                if self.checkpoint_path:
                    self.checkpoint_model(step_count=step_count)
                    if self.priority_sample:
                        self.priority_queue.save_queue(step_count=step_count)
                if step_count % self.visualisation_frequency == 0:
                    vis = True
                else:
                    vis = False
                self.validate(step_count=step_count, visualise=vis)
            self.outer_training_loop(step_count)
            print("Time taken for one step: {}".format(time.time() - t0))

    def validate(self, step_count: int, visualise: bool=True) -> None:
        """
        Performs a validation step for loss during training

        :param step_count: number of steps in training undergone (used for pring statement)
        :param visualise: whether or not to visualise validation run
        """

        validation_losses = []
        validation_figures = []
        validation_accuracies = []

        validation_parameter_tuples, validation_tasks = self._get_validation_tasks()

        for r, val_task in enumerate(validation_tasks):

            # initialise list of model iterations (used for visualisation of fine-tuning)
            validation_model_iterations = []

            # make copies of outer network for use in validation
            validation_network = copy.deepcopy(self.model_outer).to(self.device)

            # sample a task for validation fine-tuning
            validation_x_batch, validation_y_batch = self._generate_batch(task=val_task, batch_size=self.validation_k)

            validation_model_iterations.append(([w.clone() for w in validation_network._weights], [b.clone() for b in validation_network._biases]))

            # inner loop update
            for _ in range(self.validation_num_inner_updates):

                # prediction of validation batch
                validation_prediction = validation_network(validation_x_batch)

                # compute loss
                validation_loss = self._compute_loss(validation_prediction, validation_y_batch)

                # find gradients of validation loss wrt inner model weights
                network_trainable_parameters = [w for w in validation_network._weights] + [b for b in validation_network._biases]
                validation_update_grad = torch.autograd.grad(validation_loss, network_trainable_parameters, create_graph=True, retain_graph=True)

                # update inner model gradients 
                for i in range(len(validation_network._weights)):
                    validation_network._weights[i] = validation_network._weights[i] - self.inner_update_lr * validation_update_grad[i]
                for j in range(len(validation_network._biases)):
                    validation_network._biases[j] = validation_network._biases[j] - self.inner_update_lr * validation_update_grad[i + j + 1]

                current_weights = [w.clone() for w in validation_network._weights]
                current_biases = [b.clone() for b in validation_network._biases]
                validation_model_iterations.append((current_weights, current_biases))
            
            # sample a new batch from same validation task for testing fine-tuned model
            test_x_batch, test_y_batch = self._generate_batch(task=val_task, batch_size=self.test_k)

            test_prediction = validation_network(test_x_batch)
            test_loss = self._compute_loss(test_prediction, test_y_batch)

            validation_losses.append(float(test_loss))

            if self.is_classification:
                test_accuracy = self._get_accuracy(test_prediction, test_y_batch)
                validation_accuracies.append(test_accuracy)

            if visualise:
                save_name = 'validation_step_{}_rep_{}.png'.format(step_count, r)
                validation_fig = self.visualise(
                    validation_model_iterations, val_task, validation_x_batch, validation_y_batch, save_name=save_name, visualise_all=self.visualise_all
                    )
                validation_figures.append(validation_fig)

        mean_validation_loss = np.mean(validation_losses)
        var_validation_loss = np.std(validation_losses)
        mean_validation_accuracies = np.mean(validation_accuracies)

        print('--- validation loss @ step {}: {}'.format(step_count, mean_validation_loss))
        self.writer.add_scalar('meta_metrics/meta_validation_loss_mean', mean_validation_loss, step_count)
        self.writer.add_scalar('meta_metrics/meta_validation_loss_std', var_validation_loss, step_count)
        if self.is_classification:
            print('--- validation accuracy @ step {}: {}'.format(step_count, mean_validation_accuracies))
            self.writer.add_scalar('meta_metrics/validation_accuracy_mean', mean_validation_accuracies, step_count)

        # generate heatmap of validation losses 
        if self.fixed_validation:

            unique_parameter_range_lens = []
            num_parameters = len(validation_parameter_tuples[0])
            for i in range(num_parameters):
                unique_parameter_range_lens.append(len(np.unique([p[i] for p in validation_parameter_tuples])))
            validation_losses_grid = np.array(validation_losses).reshape(tuple(unique_parameter_range_lens))

            fig = plt.figure()
            plt.imshow(validation_losses_grid)
            plt.colorbar()
            
            self.writer.add_figure("validation_losses", fig, step_count)

        if visualise:
            for f, fig in enumerate(validation_figures):
                self.writer.add_figure("vadliation_plots/repeat_{}".format(f), fig, step_count)
        if self.priority_sample:
            priority_queue_fig = self.priority_queue.visualise_priority_queue()
            priority_queue_count_fig = self.priority_queue.visualise_sample_counts()
            priority_queue_loss_dist_fig = self.priority_queue.visualise_priority_queue_loss_distribution()
            self.writer.add_figure("priority_queue", priority_queue_fig, step_count)
            self.writer.add_figure("queue_counts", priority_queue_count_fig, step_count)
            self.writer.add_figure("queue_loss_dist", priority_queue_loss_dist_fig, step_count)

    def _get_validation_tasks(self):
        """produces set of tasks for use in validation"""
        if self.fixed_validation:
            return self._get_fixed_validation_tasks()
        else:
            return None, [self._sample_task() for _ in range(self.validation_task_batch_size)]

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
        torch.save({
            'step': step_count,
            'model_state_dict': self.model_outer.state_dict(),
            'optimizer_state_dict': self.meta_optimiser.state_dict(),
            }, PATH)

    @abstractmethod
    def visualise(self) -> None:
        """
        Allow for visualisation of test case. 
        E.g. a function plot for regression or a rollout for RL
        """
        raise NotImplementedError("Base class abstract method")

        