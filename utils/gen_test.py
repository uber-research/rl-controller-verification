##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'gen.py'
"""

import unittest
import utils.gen
import yaml


_MODEL_YAML = '''
  model:
    name: ppo
    load:
      value: {load_val}
      checkpoint_base_path: bazel_mount_input
      checkpoint_id: {load_checkpoint_id}
    policy:
      value: mlp
      type:
        value: custom
        layers: [16, 16]
'''


class TestGen(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def setUp(self):
        self.model = yaml.safe_load(_MODEL_YAML.format(
            load_val='False',
            load_checkpoint_id=0
        ))['model']

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_read_model(self):
        m = utils.gen.ModelDict(self.model)
        self.assertIn(m.get_model_name(), ["ddpg", "ppo"])
        self.assertEqual(m.get_actor_feature_extractor_name(), "mlp")
        self.assertEqual(m.get_actor_feature_extractor_type(), "custom")
        self.assertIsInstance(m.get_is_load(), bool)


if __name__ == '__main__':
    unittest.main()
