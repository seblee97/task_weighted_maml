from context import utils, jax_maml

import unittest

import yaml
import time 
import datetime 
import numpy as np

from typing import Any, Dict, List, Tuple

TEST_BASE_CONFIG_PATH = "test_configs/test_base_config.yaml"
TEST_CONFIG_PATH = "test_configs/test_maml_config.yaml"

# base parameters common to all configs
with open(TEST_BASE_CONFIG_PATH, 'r') as base_yaml_file:
    base_params = yaml.load(base_yaml_file, yaml.SafeLoader)

# specific parameters
with open(TEST_CONFIG_PATH, 'r') as yaml_file:
    specific_params = yaml.load(yaml_file, yaml.SafeLoader)

maml_parameters = utils.parameters.MAMLParameters(base_params) # create object in which to store experiment parameters

# update base maml parameters with specific parameters
maml_parameters.update(specific_params)

exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
experiment_name = maml_parameters.get("experiment_name")
if experiment_name:
    checkpoint_path = 'results/{}/{}/'.format(exp_timestamp, experiment_name)
else:
    checkpoint_path = 'results/{}/'.format(exp_timestamp)

maml_parameters.set_property("checkpoint_path", checkpoint_path)
maml_parameters.set_property("experiment_timestamp", exp_timestamp)

class dummyPriorityQueue(utils.priority.PriorityQueue):

    def __init__(self, 
                block_sizes: Dict[str, float], param_ranges: List[Tuple[float, float]], 
                sample_type: str, epsilon_start: float, epsilon_final: float, epsilon_decay_rate: float, epsilon_decay_start: int,
                queue_resume: str, counts_resume: str, save_path: str, burn_in: int=None, initial_value: float=None
                ):

        # convert phase bounds/ phase block_size from degrees to radians
        phase_ranges = [
            param_ranges[1][0] * (2 * np.pi) / 360, param_ranges[1][1] * (2 * np.pi) / 360
            ]
        phase_block_size = block_sizes[1] * (2 * np.pi) / 360

        param_ranges[1] = phase_ranges
        block_sizes[1] = phase_block_size
        
        super().__init__(
            block_sizes=block_sizes, param_ranges=param_ranges, sample_type=sample_type, epsilon_start=epsilon_start,
            epsilon_final=epsilon_final, epsilon_decay_rate=epsilon_decay_rate, epsilon_decay_start=epsilon_decay_start, queue_resume=queue_resume,
            counts_resume=counts_resume, save_path=save_path, burn_in=burn_in, initial_value=initial_value
        )

    def visualise_priority_queue(self, feature='losses'):
        """
        Produces plot of priority queue (losses or counts) 

        Discrete vs continuous, 2d heatmap vs 3d.

        :param feature: which aspect of queue to visualise. 'losses' or 'counts'
        :retrun fig: matplotlib figure showing heatmap of priority queue feature
        """
        pass

    def visualise_priority_queue_loss_distribution(self):
        """
        Produces probability distribution plot of losses in the priority queue
        """
        pass


class TestPriorityQueue(unittest.TestCase):

    def setUp(self):
        self.spq = dummyPriorityQueue(
                            queue_resume=maml_parameters.get(["resume", "priority_queue"]),
                            counts_resume=maml_parameters.get(["resume", "queue_counts"]),
                            sample_type=maml_parameters.get(["priority_queue", "sample_type"]),
                            block_sizes=maml_parameters.get(["priority_queue", "block_sizes"]),
                            param_ranges=maml_parameters.get(["priority_queue", "param_ranges"]),
                            initial_value=maml_parameters.get(["priority_queue", "initial_value"]),
                            epsilon_start=maml_parameters.get(["priority_queue", "epsilon_start"]),
                            epsilon_final=maml_parameters.get(["priority_queue", "epsilon_final"]),
                            epsilon_decay_start=maml_parameters.get(["priority_queue", "epsilon_decay_start"]),
                            epsilon_decay_rate=maml_parameters.get(["priority_queue", "epsilon_decay_rate"]),
                            burn_in=maml_parameters.get(["priority_queue", "burn_in"]),
                            save_path=maml_parameters.get("checkpoint_path")
                            )

    def tearDown(self):
        self.spq = None

    def test_sample_pdf_query_2d(self):
        """
        For use in case of 2d parameter space

        This test ensures that when sampling under a probability distribution given by the losses of the priority queue
        that the resulting sample is indeed proportional to the losses.
        """
        query_indices = np.arange(10 * 10)
        self.spq.sample_type = 'sample_under_pdf'

        # give priority queue a dunmmy distribution 
        dummy_distribution = np.random.random((10, 10))
        self.spq.queue = dummy_distribution

        query_samples = []

        for _ in range(10000):
            query_samples.append(self.spq.query(step=0)[0])

        import pdb; pdb.set_trace()

        return None

    def test_sample_pdf_query_3d(self):
        """
        For use in case of 3d parameter space

        This test ensures that when sampling under a probability distribution given by the losses of the priority queue
        that the resulting sample is indeed proportional to the losses.
        """
        query_indices = np.arange(10 * 10 * 10)
        self.spq.sample_type = 'sample_under_pdf'

        # give priority queue a dunmmy distribution 
        dummy_distribution = np.random.random((10, 10, 10))
        self.spq.queue = dummy_distribution

        return None

if __name__ == '__main__':
    test_cases = (TestPriorityQueue,)
    suite = unittest.TestSuite()
    for test_class in test_cases:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
    unittest.TextTestRunner(verbosity=2).run(suite)