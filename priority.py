

def get_parameter(self):
    """
    queries priority queue and returns value based on priority heuristic. 
    if max, highest value is returned
    if epsilon_greedy, highest value is return with probability 1- epsilon, else random value is returned
    if sample_interpolation, a sample is made according to a pdf defined by a continuous interpolation of the param space

    :return value: value from priority queue 
    """
    if self.sample_type == 'max':
        raise NotImplementedError
        return max(priority_queue)
    elif self.sample_type == 'epislon_greedy'
        raise NotImplementedError
        if random.random() < self.epsilon:
            return random(priority_queue)
        else:
            return max(priority_queue)
    elif self.sample_type == 'sample_interpolation':
        raise NotImplementedError
        self.priority_queue.sample()
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