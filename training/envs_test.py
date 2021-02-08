import unittest
import yaml
import numpy as np
from utils.gym_gen import f_fwgym_get_env
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import GymEnvBase


class TestEnvs(unittest.TestCase):
    """ This class regroups the unit tests
    """
    def setUp(self):
        """Prepare the test environment
        """
        with open('run/config/default.yaml', 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(self.base_config['filenames']['config_training'], 'r') as f:
            self.training_config = yaml.safe_load(f)['training']
        self.testing_params = self.training_config['training_params']
        self.is_continuous = (self.testing_params['query_generation']['value'] == "continuous")
        self.env_id = 'gym_quadcopter:quadcopter-v0'
        self.env = f_fwgym_get_env(
            env_id=self.env_id, used_states=['e_p', 'e_q', 'e_r'],
            instance_index=0, query_class='something',
            query_classes={}, params=self.testing_params
        )


    def test_env0_continuous(self):
        """ Testing the instantiation of EnvID=0
        """
        env = self.env
        self.assertEqual(env.params, self.testing_params)
        env.reset()
        self.assertEqual(env.env_id, self.env_id)
        self.assertEqual(env.instance_index, 0)
        self.assertEqual(env.continuous, self.is_continuous)


    def test_env_n_state_space_size(self):
        """ Testing Env0 State Space Size
        """
        state_space_size = {'gym_quadcopter:quadcopter-v0': 3}
        env = self.env
        self.assertEqual(env.params, self.testing_params)
        env.reset()
        self.assertEqual(env.state.shape[0], state_space_size[self.env_id], msg=f"Error in {self.env_id}")
        self.assertEqual(env.state.shape[0], env.observation_space.high.shape[0], msg=f"Error in {self.env_id}")


    def test_env_0_step(self):
        """ Testing Env0 State Space Size
        """
        action = np.array([1.0, 2.0, 3.0])
        env = self.env
        self.assertEqual(env.params, self.testing_params)
        env.reset()
        state, _, _, _ = env.step(action=action)
        #TODO do some actual test with state
        #self.assertTrue((np.array(state)[-3:] == action).all())


    def test_env_make(self):
        """ Testing Env0 State Space Size
        """
        instance_index = 0
        action = np.array([1.0, 2.0, 3.0])
        env_id = 'gym_quadcopter:quadcopter-v2'

        env1 = f_fwgym_get_env(
            env_id=env_id, used_states = ['e_p', 'e_q', 'e_r'],
            instance_index=instance_index, query_class='something',
            query_classes={}, params=self.testing_params
        )
        env1.reset()
        self.assertEqual(env1.params, self.testing_params)

        quadcopter = Quadcopter(T=20, dt_commands=0.03, dt=0.01)
        env2 = GymEnvBase.make(
            env_id=env_id, instance_index=instance_index,
            params=self.testing_params, quadcopter=quadcopter,
            used_states=['e_p', 'e_q', 'e_r']
        )
        env2.reset()
        self.assertEqual(env2.params, self.testing_params)
        self.assertEqual(env1.env_id, env2.env_id)
        self.assertEqual(env1.instance_index, env2.instance_index)
        self.assertEqual(env1.continuous, env2.continuous)


if __name__ == '__main__':
    unittest.main()
