import numpy as np 

from typing import List

def sample_nd_array(nd_array: np.array) -> [List[int], float]:
    """
    For a given n-dimensional array, this method returns an index of
    the array with probability in proportion to it's value.

    param nd_array: n-dimensional numpy array

    return indices: the index of the array
    return probability: the probability that this index was chosen
    """
    normalised_probabilities = nd_array / np.sum(nd_array)
    normalised_flattened_probabilities = normalised_probabilities.flatten()

    sample_probability = np.random.choice(normalised_flattened_probabilities, p=normalised_flattened_probabilities)

    if len(nd_array.shape) == 1:
        indices = np.where(normalised_probabilities == sample_probability)[0]
        if len(indices) > 1:
            # case of duplicate entries
            random_choice = np.random.randint(len(indices))
            indices = [indices[random_choice]]
    else:
        indices = np.where(normalised_probabilities == sample_probability)
        if len(indices) > 1:
            # case of duplicate entries
            random_choice = np.random.randint(len(indices[0]))
            indices = [int(i[random_choice]) for i in indices]

    return indices, sample_probability
