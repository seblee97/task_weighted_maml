import operator
import random
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

class PriorityQueue(object):

    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: Dict[str, Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float,
                initial_value: float=None, burn_in: int=None
                ): 
        self.block_sizes = block_sizes
        self.param_ranges = param_ranges
        self.sample_type = sample_type
        self.initial_value = initial_value
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_rate = epsilon_decay_rate 
        self.burn_in = burn_in

        self.queue = self._initialise_queue()

    def _initialise_queue(self):
        """
        create a meshgrid of dimension equal to block_sizes (number of parameters specifying task)
        for each tuple combination of parameters initialise key in queue dictionary.

        :return parameter_grid: a numpy array of dimension equal to number of parameters specifying task. 
                                initialised to a vlue specified in init
        """
        pranges = []
        for i in range(len(self.param_ranges)):
            pranges.append(int((self.param_ranges[i][1] - self.param_ranges[i][0]) / self.block_sizes[i]))

        if self.initial_value:
            parameter_grid = self.initial_value * np.zeros(tuple(pranges))
        else:
            parameter_grid = np.abs(np.random.normal(0, 1, tuple(pranges)))
        return parameter_grid
  
    # for checking if the queue is empty 
    def isEmpty(self):
        return len(self.queue) == 0
  
    # for inserting an element in the queue
    def insert(self, key, data):
        # in case of queue being a dictionary, 'key' is a key into dict object
        if type(self.queue) == dict:
            self.queue[key] = data.cpu().detach().numpy()
        # in case of queue being a np array, 'key' is a list specifying indices
        elif type(self.queue) == np.ndarray:
            self.queue[tuple(key)] = data.cpu().detach().numpy()
  
    # for popping an element based on priority heuristic
    def query(self):
        """
        queries priority queue and returns value based on priority heuristic. 
        if max, highest value is returned
        if epsilon_greedy, highest value is return with probability 1-epsilon, else random value is returned
        if sample_interpolation, a sample is made according to a pdf defined by a continuous interpolation of the param space

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
                        max_indices = np.where(self.queue == np.amax(self.queue))
                        return list(np.concatenate([list(i) for i in max_indices]))
                elif self.sample_type == 'sample_interpolation':
                    raise NotImplementedError("Currently not supported")
                else:
                    raise ValueError("No sample_type named {}. Please try either 'max', 'epsilon_greedy' or 'sample_interpolation'".format(self.sample_type))

        max_indices = get_max_indices()
        
        parameter_values = [i + b * random.random() for (i, b) in zip(max_indices, self.block_sizes)]

        # anneal epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decay_rate

        return max_indices, parameter_values, self.epsilon

    def visualise_priority_queue(self):
        """
        Produces plot of priority queue. 

        Discrete vs continuous, 2d heatmap vs 3d.
        """
        if type(self.queue) == np.ndarray:
            if len(self.queue.shape) == 2:
                fig = plt.figure()
                plt.imshow(self.queue)
                plt.colorbar()
                return fig
            else:
                raise ValueError("Visualisation with parameter space dimension > 2 not supported")
        else:
            raise NotImplementedError("Visualisation for dictionary queue not implemented")


    def interpolate_discrete_queue(self):
        """
        Make a continuous interpolation of some k-dimensional, discrete parmater queue
        """
        raise NotImplementedError
