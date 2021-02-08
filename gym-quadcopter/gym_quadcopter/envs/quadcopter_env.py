import logging
import gym
import math
import numpy as np
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import GymEnvBase, Rewards, sample
from modules.query_generator.smooth_query_generator import QueryClass
import pdb
import random as rd

class QuadcopterEnv(GymEnvBase):
    """Gym Env

    - The Env Registration ID identifies only an ID type
    - The Full Env ID is defined by the Env Registration ID and the Instance Index

    Observation Space Dim   : 13
    - Speed in 3D           : 03
    - Absolute Angles       : 03
    - Angular Rates - Querie: 03
    - Cmd                   : 04

    Attributes:
        env_id          : Env Registration ID
        instance_index  : The Index of the Instance in the Vectorized Env Data Structure
        quadcopter      : The Quadcopter Model used for interacting with the env

    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()


        # Now Quadcopter gets passed from the external 
        #self.init(env_id=0, instance_index=0, quadcopter=None, continuous=False)

        
        np.set_printoptions(precision=3)

    @property
    def old_query(self):
        return self._old_query


    @property
    def query_values(self):
        return self._query_values


    @property
    def query_classes(self):
        return self._query_classes

    @property
    def query_class(self):
        return self._query_class

    def set_query_classes(self, query_classes):
        self._query_classes = QueryClass.make_from(query_classes)

    def set_query_class(self, query_class):
        self._query_class = query_class

    def step(self, action):
        self._iterations_since_last_reset += 1
        self._commands = np.zeros(4)
        self._commands = self.f_compute_commands(action=action)
        self._commands = self.commands * self.normalize
        pwm = np.rint(np.dot(self.quadcopter.cmds_to_pwm, self.commands))
        pwm = np.minimum(pwm, self.quadcopter.saturation)
        pwm = np.maximum(pwm, 0)
        self.quadcopter.w = self.quadcopter.pwm_to_w(pwm)
        done = self.quadcopter.next_step(self.commands)
        if self.continuous:
            self.random_query()
        self._state = self._f_compute_state(cmd=self.commands/self.normalize)
        reward = self.f_reward(self.query - self.quadcopter.state[7:])

        self.event_on_step(action=action, commands=self.commands, state=self.state, reward=reward, done=done)
        
        return self.state, reward, done, {}

    def set_continuous(self,quadcopter):
        self._quadcopter = quadcopter
        self.reset()
        self._continuous = True
        logging.debug(f"set_continuous(): continuous={self.continuous}")

    def set_episodic(self,quadcopter):
        self._quadcopter = quadcopter
        self.reset()
        self._continuous = False
        logging.debug(f"set_episodic(): continuous={self.continuous}")


    def _f_compute_state(self, cmd=np.zeros(4)): 
        """ The Env Specific State Space Definition 
        """
        qs = np.zeros(10)
        qw = np.zeros(4)
        q_sat = [1.]
        pwm_1 = [50000.]
        if self.quadcopter is not None:
            qs = self.quadcopter.state
            qw = self.quadcopter.windgust
            q_sat = [self.quadcopter.saturation_motor]
            pwm_1 = [self.quadcopter.pwm_1]
        all_states = np.concatenate([
            qs[1:7],
            qs[7:],
            self.query,
            self.query - qs[7:],
            cmd,
            qw,
            q_sat,
            pwm_1
        ])
        return all_states[self.mask]

    def reset(self):
        """ Environment State Reset called by Gym Env 
        """
        self._iterations_since_last_reset = 0
        self.quadcopter.w = []
        if self.quadcopter is not None:
            # Quadcopter State Reset
            self.quadcopter.reset(params=self.params)
        self._query_memory = {
            "rate_queries": [np.zeros(3)],
            "t": [0.0]
        }
        self._query_sample = None
        if self.continuous:
            if self.query_class in self.query_classes.keys():
                self._query_sample = self.query_classes[self.query_class].sample(self.quadcopter.T, self.quadcopter.dt_commands, self.is_pwm_1)
            self.random_query()
        else :
            if self.old_query:
                self._query =  np.random.uniform(-0.6, 0.6, 3)
            else:
                value = rd.random() * 1.2 - 0.6
                if self.is_pwm_1:
                    axis = rd.randint(1, 2)
                else:
                    axis = rd.randint(1, 3)
                self._query = np.array([value] * 3) * np.isin([1, 2, 3], [axis])


        self.altitude_controller.reset()
        if self.is_pid_rates:
            self.pid_rates.reset(np.zeros(3))
        self._state = self._f_compute_state()
        self.event_on_reset()
        return self.state

    def random_query(self):
        self._query = np.random.uniform(-0.6, 0.6, 3)
        if self._query_sample is not None:
            self._query = [self._query_sample[i](self.quadcopter.t) for i in range(len(self._query_sample))]
        self._query_memory['rate_queries'].append(self.query)
        self._query_memory['t'].append(self.quadcopter.t)

    def get_reward(self, state):
        """Uses current pose of sim to return reward."""
        # return 1 + np.tanh(1 - np.abs(self.quadcopter.state[7:]).sum() / 6 * math.pi)
        # reward = 1. + max(-1., -np.abs(self.quadcopter.state[4:7]).sum() / math.pi)
        # return 1. - np.abs(self.quadcopter.state[4:7]).sum() / (math.pi)
        return max(-1., -np.abs(state[6:9]).sum() / (3 * math.pi))

    def render(self, mode='human', close=False):
        return

    def close(self):
        return
