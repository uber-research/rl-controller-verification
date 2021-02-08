import gym
import math
import numpy as np
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import GymEnvBase

class QuadcopterEnv_3_3_3_4(GymEnvBase):
    """ Generic Environment of Observation Space 3,3,3,4 
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.quadcopter = Quadcopter(T=1., dt=0.01, dt_commands=0.03)
        self.target_z = 0.
        self.k_z = [3000., 300., 2000.]
        self.z_integral = 0.
        self.previous_z_error = 0.
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -30.,
                -30.,
                -30.,
                -math.pi,
                -math.pi,
                -math.pi,
                -5 * np.pi,
                -5 * np.pi,
                -5 * np.pi,
                -5.,
                -10.,
                -1.,
                -1.,
                -1.
            ]),
            high=np.array([
                30.,
                30.,
                30.,
                math.pi,
                math.pi,
                math.pi,
                5 * np.pi,
                5 * np.pi,
                5 * np.pi,
                5.,
                10.,
                1.,
                1.,
                1.
            ])
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1., -1., -1.]),
            high=np.array([1., 1., 1.]),
        )

        self.normalize = [65535., 400., 400., 3000.]
        self.reset()
        self.continuous = False

    def step(self, action):
        commands = np.zeros(4)
        commands[1:] = action
        commands[0] = self.compute_thrust()
        commands = commands * self.normalize
        done = self.quadcopter.next_step(commands)

        self.query_duration -= self.quadcopter.dt_commands
        if self.continuous and not done and self.query_duration <= 0:
            self.random_query()

        next_state = np.concatenate([
            self.quadcopter.state[1:7],
            self.query - self.quadcopter.state[7:],
            [self.quadcopter.state[0], self.z_integral],
            commands[1:] / self.normalize[1:]
        ])

        reward = self.get_reward(next_state)

        return next_state, reward, done, {}

    def set_dt(self, dt_commands, dt):
        self.quadcopter = Quadcopter(T=1., dt_commands=dt_commands, dt=dt)

    def set_continuous(self):
        self.quadcopter = Quadcopter(T=20., dt_commands=0.01, dt=0.001)
        self.reset()
        self.continuous = True

    def reset(self):
        self.quadcopter.reset()
        self.query = np.random.uniform(-0.4, 0.4, 3)
        self.query_active = True
        self.query_duration = np.random.uniform(0.1, 1.0)
        self.query_memory = {
            "rate_queries": [np.zeros(3), self.query, self.query],
            "t": [0.0, 0.0, self.query_duration]
        }
        state = np.concatenate([
            self.quadcopter.state[1:7],
            self.query - self.quadcopter.state[7:],
            [self.quadcopter.state[0]],
            np.zeros(4)
        ])
        self.z_integral = 0.
        self.previous_z_error = 0.
        return state

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

    def f_clw_set_query(self, query, duration): 
        if duration < 0.1 or duration > 1.0: raise RuntimeError(f"Invalid Query Duration: {duration} needs to be > 0.1 and < 1.0")
        self.query_duration = duration
        self.query = query 
        self.query_memory['rate_queries'].append(np.copy(self.query))
        self.query_memory['rate_queries'].append(np.copy(self.query))
        self.query_memory['t'].append(self.quadcopter.t)
        self.query_memory['t'].append(self.quadcopter.t + self.query_duration)
        self.query_active = not self.query_active


    def get_reward(self, state):
        """Uses current pose of sim to return reward."""
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
        c = (
            self.k_z[0] * z_error + self.k_z[1] * self.z_integral
        ) + self.k_z[2] * z_derivative + 48500
        c = max(0., min(c, 65535.)) / 65535.
        return c

    def close(self):
        return
