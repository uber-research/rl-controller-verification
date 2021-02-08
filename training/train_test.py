##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'train.py'
"""

import unittest
import yaml
import numpy as np
from training.train import Training
from training.parsers import training_args_from_configs
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import Rewards
from utils.gen import EnvDict, TrainingParamsDict
from utils.gym_gen import f_fwgym_get_env


class TestTrain(unittest.TestCase):
    """ This class regroups the unit tests
    """
    def setUp(self):
        """Prepare the test environment
        """
        with open('run/config/default.yaml', 'r') as f:
            self.base_config = yaml.safe_load(f)
        with open(self.base_config['filenames']['config_training'], 'r') as f:
            self.training_config = yaml.safe_load(f)['training']
            # Overwrite some values for the unit tests environment
            self.training_config['n_steps'] = 10
            self.training_config['iterations_checkpoint'] = 10
        self.args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)
        self.supported_envs = ['0']
        self.supported_models = ['ddpg', 'ppo', 'trpo', 'td3', 'sac']
        self.reset_policy = 'abs_z velocity_x velocity_y velocity_z abs_roll abs_pitch abs_yaw rate_roll rate_pitch rate_yaw'.split(' ')


    def test_quadcopter_insantiation_default(self):
        Quadcopter()


    def test_gamma_default_value(self):
        """ This tests the gamma default value for the dppg, ppo and trpo is 0.99 by default
        """
        for model_name in self.supported_models:
            for env_id in self.supported_envs:
                print(f"-------- TESTING Model={model_name} in EnvID={env_id} ------------")
                self.training_config['model']['name'] = model_name
                args = training_args_from_configs(
                    base_config=self.base_config,
                    config=self.training_config,
                    debug_info=False
                )
                job = Training()
                res = job.run_training(args)
                self.assertEqual(job.model.gamma, 0.99)
                print(f"--------- SUCCESS Model={model_name} in EnvID={env_id} ---------")


    def test_quadcopter_reset(self):
        init_saturation = np.ones(4) * 65535.0
        q = Quadcopter(init_saturation=init_saturation)
        roll = 0.1
        pitch = -0.2
        yaw = 0.3
        q.next_step(commands=np.ones(4))
        q.reset(state=q.build_state(roll=roll, pitch=pitch, yaw=yaw))
        self.assertEqual(q.velocity_x, 0.0)
        self.assertEqual(q.velocity_y, 0.0)
        self.assertEqual(q.velocity_z, 0.0)
        self.assertEqual(q.abs_roll, roll)
        self.assertEqual(q.abs_pitch, pitch)
        self.assertEqual(q.abs_yaw, yaw)
        self.assertTrue(np.all(q.saturation == init_saturation))
        self.assertEqual(q.t, 0.0)


    def test_reward_1(self):
        """ Testing Reward on a given set of points
        """
        data = [
            {'in': np.zeros(3), 'out': 0.0},
            {'in': np.ones(3), 'out': -0.3183}
        ]
        for e in data:
            self.assertAlmostEqual(Rewards.reward_1(e['in']), e['out'], places=3)


    def test_reward_2(self):
        """ Testing Reward on a given set of points
        """
        data = [
            {'in': np.zeros(3), 'out': 1.7616}
        ]
        for e in data:
            self.assertAlmostEqual(Rewards.reward_2(e['in']), e['out'], places=3)


    def test_reward_3(self):
        """ Testing Reward on a given set of points
        """
        data = [
            {'in': np.zeros(3), 'out': 1.0}
        ]
        for e in data:
            self.assertAlmostEqual(Rewards.reward_3(e['in']), e['out'], places=3)


    def test_reward_4(self):
        """ Testing Reward on a given set of points
        """
        data = [
            {'in': np.zeros(3), 'out': 1.0}
        ]
        for e in data:
            self.assertAlmostEqual(Rewards.reward_4(e['in']), e['out'], places=3)


    def test_env(self):
        env_desc = EnvDict(env_dict=self.args.env)
        tp_desc = TrainingParamsDict(tp_dict=self.args.training_params)
        self.assertEqual(env_desc.get_env_id(), self.training_config['env']['value'])
        self.assertEqual(self.args.model, self.training_config['model'])
        self.assertEqual(self.args.n_steps, self.training_config['n_steps'])
        self.assertEqual(self.args.training_params, self.training_config['training_params'])

        env_id = 'gym_quadcopter:quadcopter-v' + str(env_desc.get_env_id())
        env = f_fwgym_get_env(
            env_id=env_id, used_states = ['e_p', 'e_q', 'e_r'],
            instance_index=0, query_class='something', query_classes={},
            params=self.args.training_params
        )
        self.assertEqual(env.params, self.args.training_params)


    def test_env_set_z_reset_function(self):
        """ Tests the possibility to set the Z Reset Function from Config Training
        """
        env_desc = EnvDict(env_dict=self.args.env)
        tp_desc = TrainingParamsDict(tp_dict=self.args.training_params)
        self.assertEqual(env_desc.get_env_id(), self.training_config['env']['value'])
        self.assertEqual(self.args.model, self.training_config['model'])
        self.assertEqual(self.args.n_steps, self.training_config['n_steps'])
        self.assertEqual(self.args.training_params, self.training_config['training_params'])

        for i in range(0,3):
            env_id = 'gym_quadcopter:quadcopter-v' + str(i)
            env = f_fwgym_get_env(
                env_id=env_id, used_states = ['e_p', 'e_q', 'e_r'],
                instance_index=0, query_class='something',
                query_classes={}, params=self.args.training_params
            )
            self.assertEqual(env.params, self.args.training_params)
            env.reset()
            val_min = float(self.args.training_params['quadcopter']['reset_policy']['abs_z']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['abs_z']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.z <= val_max)


    def test_set_saturation(self):
        """Test the saturation of the motors
        """
        env_id = 'gym_quadcopter:quadcopter-v0'
        env = f_fwgym_get_env(
            env_id=env_id, instance_index=0,
            query_class='something', query_classes={},
            params=self.args.training_params, used_states=['e_p', 'e_q', 'e_r']
        )
        self.assertSequenceEqual(
            list(env.quadcopter.saturation),
            list([self.args.training_params['quadcopter']['saturation_motor']*65535.,65535.,65535.,65535.])
        )


    def test_env_set_z_velocity_angles_reset_function(self):
        """ Tests the possibility to set the Velocity Reset Function from Config Training
        """
        env_desc = EnvDict(env_dict=self.args.env)
        tp_desc = TrainingParamsDict(tp_dict=self.args.training_params)
        self.assertEqual(env_desc.get_env_id(), self.training_config['env']['value'])
        self.assertEqual(self.args.model, self.training_config['model'])
        self.assertEqual(self.args.n_steps, self.training_config['n_steps'])
        self.assertEqual(self.args.training_params, self.training_config['training_params'])

        for i in range(0,3):
            env_id = f'gym_quadcopter:quadcopter-v{i}'
            env = f_fwgym_get_env(
                env_id=env_id, used_states = ['e_p', 'e_q', 'e_r'],
                instance_index=0, query_class='something',
                query_classes={}, params=self.args.training_params
            )
            self.assertEqual(env.params, self.args.training_params)

            env.reset()
            val_min = float(self.args.training_params['quadcopter']['reset_policy']['abs_z']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['abs_z']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.z <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['velocity_x']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['velocity_x']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.velocity_x <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['velocity_y']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['velocity_y']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.velocity_y <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['velocity_z']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['velocity_z']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.velocity_z <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['abs_roll']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['abs_roll']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.abs_roll <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['abs_pitch']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['abs_pitch']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.abs_pitch <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['abs_yaw']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['abs_yaw']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.abs_yaw <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['rate_roll']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['rate_roll']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.rate_roll <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['rate_pitch']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['rate_pitch']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.rate_pitch <= val_max)

            val_min = float(self.args.training_params['quadcopter']['reset_policy']['rate_yaw']['params'][0])
            val_max = float(self.args.training_params['quadcopter']['reset_policy']['rate_yaw']['params'][1])
            self.assertTrue(val_min <= env.quadcopter.rate_yaw <= val_max)


    def test_training_base(self):
        """ Runs a full training only few iterations long as a sort of integration test
        """
        for model_name in self.supported_models:
            for env_id in self.supported_envs:
                print(f"-------- TESTING Model={model_name} in EnvID={env_id} ------------")
                self.training_config['model']['name'] = model_name
                self.training_config['env']['value'] = env_id
                self.training_config['n_steps'] = 10
                self.training_config['iterations_checkpoint'] = 10
                args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)
                job = Training()
                res = job.run_training(args)
                self.assertTrue(res)
                print(f"--------- SUCCESS Model={model_name} in EnvID={env_id} ---------")


    def test_logging_base(self):
        """ Runs a full training only few iterations long as a sort of integration test
        """
        self.supported_models = ['ddpg']    #TODO find out why the test times out if we don't limit it to ddpg
        for model_name in self.supported_models:
            for env_id in self.supported_envs:
                self.training_config['model']['name'] = model_name
                self.training_config['env']['value'] = env_id
                self.training_config['n_steps'] = 0

                self.training_config['logging']['framework']['log_filename']['active'] = True
                args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)
                job = Training()
                res = job.run_training(args)

                self.training_config['logging']['framework']['log_filename']['active'] = False
                args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)
                job = Training()
                res = job.run_training(args)


    def test_training_reset_policy_none(self):
        """ Runs a full training only few iterations long as a sort of integration test to show T6469163 is fixed
        """
        for model_name in self.supported_models:
            for env_id in self.supported_envs:
                self.training_config['model']['name'] = model_name
                self.training_config['env']['value'] = env_id
                self.training_config['n_steps'] = 10
                self.training_config['iterations_checkpoint'] = 10
                print(f"-------- TESTING Model={model_name} in EnvID={env_id} ------------")
                args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)
                job = Training()
                res = job.run_training(args)
                self.assertTrue(res)
                print(f"--------- SUCCESS Model={model_name} in EnvID={env_id} ---------")


    def test_env_altitude_controller_temporal_consistency(self):
        """ Tests that env.altitude_controller.compute_thrust() has no random component
        """
        for pol in self.reset_policy:
            self.training_config['training_params']['quadcopter']['reset_policy'][pol]['pdf'] = 'none'
        self.training_config['model']['name'] = 'ddpg'
        self.args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)

        env_desc = EnvDict(env_dict=self.args.env)
        tp_desc = TrainingParamsDict(tp_dict=self.args.training_params)
        self.assertEqual(env_desc.get_env_id(), self.training_config['env']['value'])
        self.assertEqual(self.args.model, self.training_config['model'])
        self.assertEqual(self.args.n_steps, self.training_config['n_steps'])
        self.assertEqual(self.args.training_params, self.training_config['training_params'])

        supported_envs = [0]
        for i in supported_envs:
            env_id = f'gym_quadcopter:quadcopter-v{i}'
            print(f"Checking EnvID={env_id}")
            env = f_fwgym_get_env(
                env_id=env_id, used_states = ['e_p', 'e_q', 'e_r'],
                instance_index=0, query_class='something',
                query_classes={}, params=self.args.training_params
            )
            self.assertEqual(env.params, self.args.training_params)

            self.assertEqual(env.altitude_controller.p, 3000)
            self.assertEqual(env.altitude_controller.i, 300)
            self.assertEqual(env.altitude_controller.d, 500)

            obs_trace = np.zeros(10)
            exp_trace = np.zeros(10)

            env.reset()
            env.set_target_z(1.0)
            self.assertEqual(env.quadcopter.z, 0.0)
            self.assertEqual(env.target_z, 1.0)

            self.assertEqual(env.altitude_controller.z_integral, 0.0)
            self.assertEqual(env.previous_z_error, 0.0)

            for i in range(obs_trace.shape[0]):
                self.assertEqual(env.quadcopter.z, 0.0)
                exp_trace[i] = env.altitude_controller.compute_thrust(target_z=env.target_z, current_z=env.quadcopter.z)

            env.reset()
            env.set_target_z(1.0)
            self.assertEqual(env.quadcopter.z, 0.0)
            self.assertEqual(env.target_z, 1.0)

            self.assertEqual(env.altitude_controller.z_integral, 0.0)
            self.assertEqual(env.previous_z_error, 0.0)

            for i in range(obs_trace.shape[0]):
                self.assertEqual(env.quadcopter.z, 0.0)
                obs_trace[i] = env.altitude_controller.compute_thrust(target_z=env.target_z, current_z=env.quadcopter.z)

            self.assertTrue(np.allclose(obs_trace, exp_trace, atol=1e-5), msg=f"Temporal Consistency Check: EnvID={env_id} ObsTrace={obs_trace}, ExpTrace={exp_trace}, Delta={obs_trace-exp_trace}")


    def test_env_controller_temporal_consistency_on_altitude(self):
        """ Tests if given the same initial conditions the and target_z the selected altitude controller leads to the same z after the same number of iterations

        WARNING
        - This test is currently failing due to some precision issues hence it has been disabled but it revealed an issue that needs to be deeply investigated hence it needs to be commited in the repo
        """
        for pol in self.reset_policy:
            self.training_config['training_params']['quadcopter']['reset_policy'][pol]['pdf'] = 'none'
        self.training_config['model']['name'] = 'ddpg'
        self.args = training_args_from_configs(base_config=self.base_config, config=self.training_config, debug_info=False)

        env_desc = EnvDict(env_dict=self.args.env)
        tp_desc = TrainingParamsDict(tp_dict=self.args.training_params)
        self.assertEqual(env_desc.get_env_id(), self.training_config['env']['value'])
        self.assertEqual(self.args.model, self.training_config['model'])
        self.assertEqual(self.args.n_steps, self.training_config['n_steps'])
        self.assertEqual(self.args.training_params, self.training_config['training_params'])

        for i in self.supported_envs:
            env_id = f'gym_quadcopter:quadcopter-v{i}'
            print(f"Checking EnvID={env_id}")
            env = f_fwgym_get_env(
                env_id=env_id, used_states = ['e_p', 'e_q', 'e_r'],
                instance_index=0, query_class='something',
                query_classes={}, params=self.args.training_params
            )
            self.assertEqual(env.params, self.args.training_params)

            self.assertEqual(env.altitude_controller.p, 3000)
            self.assertEqual(env.altitude_controller.i, 300)
            self.assertEqual(env.altitude_controller.d, 500)

            obs_trace = np.zeros(10)
            exp_trace = np.zeros(10)

            env.reset()
            env.set_target_z(1.0)
            self.assertEqual(env.quadcopter.z, 0.0)
            self.assertEqual(env.target_z, 1.0)
            self.assertEqual(env.z_integral, 0.0)
            self.assertEqual(env.previous_z_error, 0.0)
            for i in range(obs_trace.shape[0]):
                env.step(action=np.zeros(3))
                exp_trace[i] = env.quadcopter.z

            env.reset()
            env.set_target_z(1.0)
            self.assertEqual(env.quadcopter.z, 0.0)
            self.assertEqual(env.target_z, 1.0)
            self.assertEqual(env.z_integral, 0.0)
            self.assertEqual(env.previous_z_error, 0.0)
            for i in range(obs_trace.shape[0]):
                env.step(action=np.zeros(3))
                obs_trace[i] = env.quadcopter.z

            self.assertTrue(np.allclose(obs_trace, exp_trace, atol=1e-5), msg=f"Temporal Consistency Check: EnvID={env_id} ObsTrace={obs_trace}, ExpTrace={exp_trace}, Delta={obs_trace-exp_trace}")


if __name__ == '__main__':
    unittest.main()
