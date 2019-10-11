import numpy as np 

from typing import List, Tuple

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


def get_convolutional_output_dims(input_shape: Tuple, output_depth: int, kernel_sizes: List[int], strides: List[int], paddings: List[int]) -> int:
    """
    For convolutional module defined by paramters given, compute number of 
    parameters in final layer (i.e. input dimension for flatten operation)

    For square inputs with equal stride/pad/kernel magnitudes in height and width
    TODO: generalise
    """
    width_height = input_shape[0]
    for layer in range(len(kernel_sizes)):
        width_height = 1 + (width_height + 2 * paddings[layer] - kernel_sizes[layer] - 2) / strides[layer]
    num_params = int(output_depth * width_height ** 2)

    return num_params