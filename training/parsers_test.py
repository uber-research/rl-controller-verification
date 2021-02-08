##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'plots.py'
"""
import unittest
import yaml
from pathlib import Path
from training.parsers import training_args_from_configs


class TestPlots(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def setUp(self):
        """Prepare the test environment
        """
        with open('run/config/default.yaml', 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(self.base_config['filenames']['config_training'], 'r') as f:
            self.training_config = yaml.safe_load(f)['training']
        self.args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)


    def test_parsing(self):
        """ Checks all the most important fields are there
        """
        self.assertEqual(self.args.env, self.training_config['env'])
        self.assertEqual(self.args.model, self.training_config['model'])
        self.assertEqual(self.args.n_steps, self.training_config['n_steps'])
        self.assertEqual(self.args.training_params, self.training_config['training_params'])


    def test_additional_tf_logs(self):
        """ Checks tensorflow logging keys
        """
        self.assertTrue('summary' in self.args.logging['tensorflow'])
        self.assertTrue('stable_baselines_tensors' in self.args.logging['tensorflow'])


    def test_parsing_testing_params(self):
        self.assertEqual(
            self.args.training_params['query_generation']['continuous']['T_episode'],
            self.training_config['training_params']['query_generation']['continuous']['T_episode']
        )


    def test_info_files(self):
        """ Checks both the config_training.yaml and the meta.yaml gets copied in the right position
        """
        self.assertTrue(Path(self.base_config['filenames']['config_training']).exists())
        self.assertTrue('meta_info_filename' in self.base_config['filenames'])


if __name__ == '__main__':
    unittest.main()
