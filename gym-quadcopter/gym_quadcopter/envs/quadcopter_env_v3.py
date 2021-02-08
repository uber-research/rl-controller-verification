import logging
import gym
import json
import math
import numpy as np
import os.path
from gym_quadcopter.envs.quadcopter_2 import Quadcopter


class QuadcopterEnv3(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env_id = 3
        self.instance_index = 0 

        DIR_NAME = os.path.dirname(__file__)
        filename = os.path.join(DIR_NAME, 'config.json')
        with open(filename) as config:
            self.params = json.load(config)["env"]

        self.quadcopter = None 
        self.target_z = 0.
        self.k_z = self.params["pid_z"]
        self.z_integral = 0.
        self.previous_z_error = 0.

        self.observation_space = gym.spaces.Box(
            low=np.array([
                -self.params["speed_max"], 
                -self.params["speed_max"], 
                -self.params["speed_max"],
                -self.params["angle_max"], 
                -self.params["angle_max"], 
                -self.params["angle_max"],
                -self.params["angular_rate_max"], 
                -self.params["angular_rate_max"], 
                -self.params["angular_rate_max"],
                -1., 
                -1., 
                -1., 
                -1., ]),  # 0., 0., 0., 0.]),
            high=np.array([
                self.params["speed_max"], self.params["speed_max"], self.params["speed_max"],
                self.params["angle_max"], self.params["angle_max"], self.params["angle_max"],
                self.params["angular_rate_max"], self.params["angular_rate_max"], self.params["angular_rate_max"],
                1., 1., 1., 1.]),  # 1., 1., 1., 1.]),
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1., -1., -1.]),
            high=np.array([1., 1., 1., 1.]),
        )

        self.normalize = self.quadcopter.params["PWM_max"]
        self.reset()
        self.continuous = False
        self.event_on_instantiation()

    def step(self, action):
        # f = self.compute_thrust()
        commands = np.rint((action + 1) * 0.5 * self.normalize)
        done = self.quadcopter.next_step(commands, True)
        self.query_duration -= self.quadcopter.dt_commands
        if self.continuous and not done and self.query_duration <= 0:
            self.random_query()

        next_state = np.concatenate([
            self.quadcopter.state[1:7],
            self.query - self.quadcopter.state[7:],
            action,
        ])
        reward = self.get_reward(next_state)

        return next_state, reward, done, {}

    def set_continuous(self, quadcopter):
        self.quadcopter = quadcopter
        self.reset()
        self.continuous = True
        logging.debug(f"set_continuous(): continuous={self.continuous}")

    def set_episodic(self, quadcopter):
        self.quadcopter = quadcopter
        self.reset()
        self.continuous = False
        logging.debug(f"set_episodic(): continuous={self.continuous}")

    def f_get_state(self): 
        """ The Env Specific State Space Definition 
        """
        if self.quadcopter is not None: 
            return np.concatenate([
                self.quadcopter.state[1:7],
                self.query - self.quadcopter.state[7:],
                np.zeros(4)
            ])
        return np.zeros(13)

    def reset(self):
        if self.quadcopter is not None: self.quadcopter.reset()
        self.query = np.random.uniform(-0.4, 0.4, 3)
        self.query_active = True
        self.query_duration = np.random.uniform(0.1, 1.0)
        self.query_memory = {
            "rate_queries": [np.zeros(3), self.query, self.query],
            "t": [0.0, 0.0, self.query_duration]
        }
        self.saturation = np.ones(4)

        # This code was already here 
        # i = np.random.randint(0, 4)
        # r = np.random.rand() / 4 + 0.75
        # self.saturation[i] = r

        # self.saturation[(i + 2) % 4] = r
        if self.quadcopter is not None: self.quadcopter.set_saturation(self.saturation * self.normalize)

        self.state = self.f_get_state()

        self.z_integral = 0.
        self.previous_z_error = 0.
        return self.state

    def random_query(self):
        if self.query_active:
            self.query_duration = np.random.uniform(0.1, 1.0)
            self.query = np.zeros(3)
        else:
            self.query_duration = np.random.uniform(0.1, 1.0)
            self.query = np.random.uniform(-0.6, 0.6, 3)
        self.query_memory['rate_queries'].append(np.copy(self.query))
        self.query_memory['rate_queries'].append(np.copy(self.query))
        self.query_memory['t'].append(self.quadcopter.t)
        self.query_memory['t'].append(self.quadcopter.t + self.query_duration)
        self.query_active = not self.query_active

    def get_reward(self, state):
        """Uses current pose of sim to return reward."""
        # This code was already here 
        # return 1 + np.tanh(1 - np.abs(self.quadcopter.state[7:]).sum() / 6 * math.pi)
        # reward = 1. + max(-1., -np.abs(self.quadcopter.state[4:7]).sum() / math.pi)
        # return 1. - np.abs(self.quadcopter.state[4:7]).sum() / (math.pi)
        return max(-1., -np.abs(state[6:9]).sum() / (3 * math.pi))

    def render(self, mode='human', close=False):
        return

    def compute_thrust(self):
        z_error = self.target_z - self.quadcopter.state[0]
        z_derivative, self.previous_z_error = (
            z_error - self.previous_z_error
        ) / self.quadcopter.dt, z_error
        self.z_integral += z_error * self.quadcopter.dt
        c = self.k_z[0] * z_error
        c += self.k_z[1] * self.z_integral
        c += self.k_z[2] * z_derivative + 48500
        c = max(0., min(c, 65535.))
        return c

    def close(self):
        return
