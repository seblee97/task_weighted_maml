from context import utils, jax_maml

import unittest

import yaml
import time 
import datetime 

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

class TestPriorityQueue(unittest.TestCase):

    def setUp(self):
        self.spq = jax_maml.jax_sinusoid.SinePriorityQueue(
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

    def test_2d_visualise_priority_queue(self):
        return None

    def test_3d_visualise_priority_queue(self):
        return None
        
    def visualise_priority_queue_loss_distribution(self):
        return None


        

if __name__ == '__main__':
    test_cases = (TestPriorityQueue,)
    suite = unittest.TestSuite()
    for test_class in test_cases:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
    unittest.TextTestRunner(verbosity=2).run(suite)