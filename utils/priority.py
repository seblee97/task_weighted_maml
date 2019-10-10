import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from scipy import stats, interpolate
import copy

from typing import List, Dict, Tuple

from abc import ABC, abstractmethod

import utils.custom_functions

class PriorityQueue(ABC):
    """
    Base class for the priority queue.
    """
    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: Dict[str, Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float, epsilon_decay_start: int,
                queue_resume: str, counts_resume: str, save_path: str, burn_in: int=None, initial_value: float=None, scale_parameters: bool=True
                ):
        self.queue_resume = queue_resume
        self.counts_resume = counts_resume
        self.block_sizes = block_sizes
        self.param_ranges = param_ranges
        self.sample_type = sample_type
        self.initial_value = initial_value
        self.scale_parameters = scale_parameters

        if 'epsilon' in self.sample_type:
            self.epsilon = epsilon_start
            self.epsilon_final = epsilon_final
            self.epsilon_decay_start = epsilon_decay_start
            self.epsilon_decay_rate = epsilon_decay_rate 
        else:
            self.epsilon = None 

        self.burn_in = burn_in
        self.save_path = save_path

        self._queue, self.sample_counts, self._queue_delta = self._initialise_queue() 

    def get_queue(self):
        """
        getter method for priority queue
        """
        return self._queue

    def get_epsilon(self):
        """
        getter method for epsilon value
        """
        return self.epsilon

    def _initialise_queue(self):
        """
        create a meshgrid of dimension equal to block_sizes (number of parameters specifying task)
        for each tuple combination of parameters, initialise key in queue dictionary either to 
        constant value or gaussian. 
        
        construct analogous grid for tracking counts of sampling of different parameter tuples. 
        construct analogous grid for tracking first derivative of original grid.

        :return parameter_grid: a numpy array of dimension equal to number of parameters specifying task. 
                                initialised to a vlue specified in init
        :return counts: a numpy array of dimension equal to number of parameters specifying task with counts
                        all set to zero (unless using saved version)
        :return queue_delta: a numpy array of dimension equal to number of parameters specifying task with change in
                             parameter_grid
        """
        if self.queue_resume:
            # load saved priority queue from previous run
            parameter_grid = np.load(self.queue_resume)
            counts = np.load(self.counts_resume)
        
        else:
            pranges = []
            for i in range(len(self.param_ranges)):
                pranges.append(int((self.param_ranges[i][1] - self.param_ranges[i][0]) / self.block_sizes[i]))

            if self.initial_value:
                parameter_grid_init = self.initial_value * np.zeros(tuple(pranges))
                parameter_grid = self.initial_value * np.zeros(tuple(pranges))
                queue_delta = parameter_grid - parameter_grid_init
            else:
                parameter_grid_init = np.abs(np.random.normal(0, 1, tuple(pranges)))
                parameter_grid = np.abs(np.random.normal(0, 1, tuple(pranges)))
                queue_delta = np.abs(np.random.normal(0, 30, tuple(pranges)))

            counts = np.zeros(tuple(pranges))

        return parameter_grid, counts, queue_delta

    def save_queue(self, step_count: int) -> None:
        """
        Save a copy of the priority_queue and queue_counts up to this point in training

        :param step_count: iteration number of training (meta-steps)
        """
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
        # format of model chekcpoint path: timestamp _ step_count
        queue_path = '{}priority_queue_{}_{}.npz'.format(self.save_path, timestamp, str(step_count))
        counts_path = '{}queue_counts_{}_{}.npz'.format(self.save_path, timestamp, str(step_count))
        np.savez(queue_path, self._queue)
        np.savez(counts_path, self.sample_counts)
  
    def insert(self, key: List, data: float) -> None:
        """
        inserts element into queue.

        :param key: index of queue to update
        :param data: new value 
        """
        current_data = self._queue[tuple(key)]
        data_delta = current_data - data
        self._queue[tuple(key)] = data
        self._queue_delta[tuple(key)] = abs(data_delta)
  
    def query(self, step: int):
        """
        queries priority queue and returns value based on priority heuristic. 
        if max, highest value is returned
        if epsilon_greedy, highest value is return with probability 1-epsilon, else random value is returned
        if interpolate_and_sample_under_pdf, a sample is made according to a pdf defined by a continuous interpolation of the param space
        if sample_under_pdf, a sample is made according to discrete pdf from counts
        if sample_delta, a sample is made accordint to discrete pdf from queue delta
        if importance is added to above, task probabilities are tracked for use in reweighting later

        :param step: step count of training

        :return indices: indices of priority queue
        :return parameter_values: values of parameters for task (obtained from indices)
        :return task_probability: probability that the task sampled was to be sampled 
        """
        queue_copy = copy.deepcopy(self._queue)

        if type(self._queue) != np.ndarray:
            raise ValueError("Incorrect type for priority queue, must be numpy array")

        if self.sample_type == 'max':
            raise NotImplementedError("Currently not supported - need a way to fill buffer before this would make sense to use")

        elif self.sample_type == 'epsilon_greedy':
            if random.random() < self.epsilon: # select randomly
                indices = [np.random.randint(d) for d in self._queue.shape]
            else: # select greedily
                max_indices = np.array(np.where(self._queue == np.amax(self._queue))).T
                if len(max_indices) > 1:
                    indices = random.choice(max_indices).tolist()
                else:
                    indices = max_indices[0].tolist()
            task_probability = 1.

        elif 'sample_under_pdf' in self.sample_type:
            indices, task_probability = utils.custom_functions.sample_nd_array(nd_array=self._queue)
            if "importance" not in self.sample_type:
                task_probability = 1.

        elif 'sample_delta' in self.sample_type:
            indices, task_probability = utils.custom_functions.sample_nd_array(nd_array=self._queue_delta)
            if "importance" not in self.sample_type:
                task_probability = 1.

        elif self.sample_type == 'interpolate_and_sample_under_pdf': # probably unnecessary if parameter grid is finely grained enough
            if len(self._queue.shape) == 2:
                # get interpolation function
                interpolated_queue = interpolate.interp2d(np.arange(self._queue.shape[0]), np.arange(self._queue.shape[1]), self._queue)
                
                # generate more finely grained dummy grid (1e6 x 1e6)
                x = np.arange(self._queue.shape[0], step=self._queue.shape[0] / 1e6)
                y = np.arange(self._queue.shape[1], step=self._queue.shape[1] / 1e6)

                z = interpolated_queue(x, y)
                # sample according to z
                raise NotImplementedError(
                    "Unfinished implementation. Sample according to z, which is more finely grained than priority queue. Use 'sample_under_pdf' sample_type"
                    "which should have similar properties with simpler implementation wrt consistency of method spec (i.e. returning priority queue indices"
                    "as well as ultimate parameter values"
                    )
            else:
                raise NotImplementedError(
                    "Currently interpolation method is only supported for priority queue dimensions of 2 i.e. for tasks specified by two parameters"
                    )

        else:
            raise ValueError("No sample_type named {}. Please try either 'max', 'epsilon_greedy', 'sample_under_pdf', or 'interpolate_and_sample_under_pdf'".format(self.sample_type))

        # add to sample count of max_indices
        if len(indices) == 1:
            self.sample_counts[indices[0]] += 1
        else:
            self.sample_counts[tuple(indices)] += 1

        # convert samples/max indices to parameter values (i.e. scale by parameter ranges)
        if self.scale_parameters:
            parameter_values = [p[0] + i * b + random.uniform(0, b) for (p, i, b) in zip(self.param_ranges, indices, self.block_sizes)]
        else:
            parameter_values = indices

        # anneal epsilon
        if self.epsilon and (self.epsilon > self.epsilon_final) and (step > self.epsilon_decay_start):
            self.epsilon -= self.epsilon_decay_rate

        assert (self._queue == queue_copy).all(), "Error. Queue should not change under query method."

        return indices, parameter_values, task_probability

    @abstractmethod
    def visualise_priority_queue(self):
        """
        Produces plot of priority queue. 

        Discrete vs continuous, 2d heatmap vs 3d.
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def visualise_priority_queue_loss_distribution(self):
        """
        Produces probability distribution plot of losses in the priority queue
        """
        raise NotImplementedError("Base class method")

    def interpolate_discrete_queue(self):
        """
        Make a continuous interpolation of some k-dimensional, discrete parmater queue
        """
        raise NotImplementedError

    def compute_count_loss_correlation(self) -> float:
        """
        computes the correlation between the number of times a range in the parameter space
        has been queried and the loss associated with that range.

        Correlation is defined as the spearman's rank correlation coefficient 
        between the flattened matrices
        """
        flattened_losses = self._queue.flatten()
        flattened_counts = self.sample_counts.flatten()

        spearmans_rank = stats.spearmanr(flattened_counts, flattened_losses).correlation

        return spearmans_rank
