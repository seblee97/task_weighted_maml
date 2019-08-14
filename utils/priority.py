import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

from typing import List, Dict, Tuple

from abc import ABC, abstractmethod

class PriorityQueue(ABC):

    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: Dict[str, Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float, epsilon_decay_start: int,
                queue_resume: str, counts_resume: str, save_path: str, burn_in: int=None, initial_value: float=None
                ):
        self.queue_resume = queue_resume
        self.counts_resume = counts_resume
        self.block_sizes = block_sizes
        self.param_ranges = param_ranges
        self.sample_type = sample_type
        self.initial_value = initial_value
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.burn_in = burn_in
        self.save_path = save_path

        self.queue, self.sample_counts = self._initialise_queue() 

    def _initialise_queue(self):
        """
        create a meshgrid of dimension equal to block_sizes (number of parameters specifying task)
        for each tuple combination of parameters initialise key in queue dictionary.

        :return parameter_grid: a numpy array of dimension equal to number of parameters specifying task. 
                                initialised to a vlue specified in init
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
                parameter_grid = self.initial_value * np.zeros(tuple(pranges))
            else:
                parameter_grid = np.abs(np.random.normal(0, 1, tuple(pranges)))

            counts = np.zeros(tuple(pranges))
        return parameter_grid, counts

    def save_queue(self, step_count):
        """
        Save a copy of the priority_queue and queue_counts up to this point in training

        :param step_count: iteration number of training (meta-steps)
        """
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H-%M-%S')
        # format of model chekcpoint path: timestamp _ step_count
        queue_path = '{}priority_queue_{}_{}.npz'.format(self.save_path, timestamp, str(step_count))
        counts_path = '{}queue_counts_{}_{}.npz'.format(self.save_path, timestamp, str(step_count))
        np.savez(queue_path, self.queue)
        np.savez(counts_path, self.sample_counts)
  
    # for checking if the queue is empty 
    def isEmpty(self):
        return len(self.queue) == 0
  
    # for inserting an element in the queue
    def insert(self, key, data):
        try:
            # in case of queue being a dictionary, 'key' is a key into dict object
            if type(self.queue) == dict:
                self.queue[key] = data.cpu().detach().numpy()
            # in case of queue being a np array, 'key' is a list specifying indices
            elif type(self.queue) == np.ndarray:    
                self.queue[tuple(key)] = data.cpu().detach().numpy()
        except:
            import pdb; pdb.set_trace()
  
    # for popping an element based on priority heuristic
    def query(self, step):
        """
        queries priority queue and returns value based on priority heuristic. 
        if max, highest value is returned
        if epsilon_greedy, highest value is return with probability 1-epsilon, else random value is returned
        if sample_interpolation, a sample is made according to a pdf defined by a continuous interpolation of the param space

        :param step: step count of training
        :return value: value from priority queue
        """
        def get_max_indices():
            if type(self.queue) == dict:
                if self.sample_type == 'max':
                    raise NotImplementedError("Currently not supported - need a way to fill buffer before this would make sense to use")
                    return max(self.queue.items(), key=operator.itemgetter(1))[0]
                elif self.sample_type == 'epsilon_greedy':
                    if random.random() < self.epsilon: # select randomly
                        return random.choice(list(self.queue.keys()))
                    else: # select greedily
                        return max(self.queue.items(), key=operator.itemgetter(1))[0]
                elif self.sample_type == 'sample_interpolation':
                    raise NotImplementedError("Currently not supported")
                else:
                    raise ValueError("No sample_type named {}. Please try either 'max', 'epsilon_greedy' or 'sample_interpolation'".format(self.sample_type))
            elif type(self.queue) == np.ndarray:
                if self.sample_type == 'max':
                    raise NotImplementedError("Currently not supported - need a way to fill buffer before this would make sense to use")
                elif self.sample_type == 'epsilon_greedy':
                    if random.random() < self.epsilon: # select randomly
                        random_indices = [np.random.randint(d) for d in self.queue.shape]
                        return random_indices
                    else: # select greedily
                        max_indices = np.array(np.where(self.queue == np.amax(self.queue))).T
                        if len(max_indices) > 1:
                            return random.choice(max_indices).tolist()
                        else:
                            return max_indices[0].tolist()
                elif self.sample_type == 'sample_interpolation':
                    raise NotImplementedError("Currently not supported")
                else:
                    raise ValueError("No sample_type named {}. Please try either 'max', 'epsilon_greedy' or 'sample_interpolation'".format(self.sample_type))
    
        indices = get_max_indices()

        # add to sample count of max_indices
        try:
            self.sample_counts[tuple(indices)] += 1
        except:
            import pdb; pdb.set_trace()
        
        # convert samples/max indices to parameter values (i.e. scale by parameter ranges)
        parameter_values = [p[0] + i * b + random.uniform(0, b) for (p, i, b) in zip(self.param_ranges, indices, self.block_sizes)]

        # anneal epsilon
        if self.epsilon > self.epsilon_final and step > self.epsilon_decay_start:
            self.epsilon -= self.epsilon_decay_rate

        return indices, parameter_values, self.epsilon

    @abstractmethod
    def visualise_priority_queue(self):
        """
        Produces plot of priority queue. 

        Discrete vs continuous, 2d heatmap vs 3d.
        """
        raise NotImplementedError("Base class method")
    
    @abstractmethod
    def visualise_sample_counts(self):
        """
        Produces plot of priority queue sampling counts 
        """
        raise NotImplementedError("Base class method")

    def interpolate_discrete_queue(self):
        """
        Make a continuous interpolation of some k-dimensional, discrete parmater queue
        """
        raise NotImplementedError

    def compute_count_loss_correlation(self):
        """
        computes the correlation between the number of times a range in the parameter space
        has been queried and the loss associated with that range.

        Correlation is defined with respect to the flattened matrices
        """
        flattened_losses = self.queue.flatten()
        flattened_counts = self.sample_counts.flatten()
        return np.correlate(flattened_losses, flattened_counts)[0]

