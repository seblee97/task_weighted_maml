import operator
import random

from typing import List, Dict

class PriorityQueue(object):

    def __init__(self, block_sizes: Dict[str, float], param_ranges: Dict[str, Tuple[float, float]], sample_type: str, burn_in: int=None): 
        self.block_sizes = block_sizes
        self.param_ranges = param_ranges
        self.sample_type = sample_type
        self.burn_in = burn_in

        self.queue = self._initialise_queue()

    def _initialise_queue(self):
        """
        create a meshgrid of dimension equal to block_sizes (number of parameters specifying task)
        for each tuple combination of parameters initialise key in queue dictionary.
        """
        raise NotImplementedError
  
    # for checking if the queue is empty 
    def isEmpty(self):
        return len(self.queue) == 0
  
    # for inserting an element in the queue 
    def insert(self, key, data): 
        self.queue[key] = data
  
    # for popping an element based on priority heuristic
    def query(self):
        """
        queries priority queue and returns value based on priority heuristic. 
        if max, highest value is returned
        if epsilon_greedy, highest value is return with probability 1-epsilon, else random value is returned
        if sample_interpolation, a sample is made according to a pdf defined by a continuous interpolation of the param space

        :return value: value from priority queue
        """
        if self.sample_type == 'max':
            raise NotImplementedError("Currently not supported - need a way to fill buffer before this would make sense to use")
            return max(self.queue.items(), key=operator.itemgetter(1))[0]
        elif self.sample_type == 'epislon_greedy'
            if random.random() < self.epsilon: # select randomly
                return random.choice(list(self.queue.keys()))
            else: # select greedily
                return max(self.queue.items(), key=operator.itemgetter(1))[0]
        elif self.sample_type == 'sample_interpolation':
            raise NotImplementedError("Currently not supported")
        else:
            raise ValueError("No sample_type named {}. Please try either 'max', 'epsilon_greedy' or 'sample_interpolation'".format(self.sample_type))

    def visualise_priority_queue(self):
        """
        Produces plot of priority queue. 

        Discrete vs continuous, 2d heatmap vs 3d.
        """
        raise NotImplementedError


    def interpolate_discrete_queue(self):
        """
        Make a continuous interpolation of some k-dimensional, discrete parmater queue
        """
        raise NotImplementedError
