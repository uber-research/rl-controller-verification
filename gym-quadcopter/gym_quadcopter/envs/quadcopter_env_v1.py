import logging
import gym
import math
import numpy as np
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import GymEnvBase

class QuadcopterEnv1(GymEnvBase):
    """Gym Env

    - The Env Registration ID identifies only an ID type
    - The Full Env ID is defined by the Env Registration ID and the Instance Index

    Observation Space Dim     : 17
    - Speed in 3D             : 03
    - Absolute Angles         : 03
    - Angular Rates - Queries : 03
    - Cmd (with the thrust)   : 04
    - Saturation Commands     : 04

    Attributes:
        env_id          : Env Registration ID
        instance_index  : The Index of the Instance in the Vectorized Env Data Structure
        quadcopter      : The Quadcopter Model used for interacting with the env

    """

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        #self.init(env_id=1, instance_index=0, quadcopter=None, continuous=False)
        self._k_z = [3000., 300., 2000.] # P - I - D

        self._observation_space = gym.spaces.Box(
            low=np.array(
                [-30., -30., -30., - math.pi, - math.pi, - math.pi, -5 * np.pi, -5 * np.pi, -5 * np.pi, 0., -1., -1., -1., 0.8, 0.8, 0.8, 0.8]  # noqa E261
            ),
            high=np.array(
                [30., 30., 30., math.pi, math.pi, math.pi, 5 * np.pi, 5 * np.pi, 5 * np.pi, 1., 1., 1., 1., 1., 1., 1., 1.]  # noqa E261
            )
        )
        self._action_space = gym.spaces.Box(
            low=np.array([-1., -1., -1.]),
            high=np.array([1., 1., 1.]),
        )

        self._saturation = np.ones(4)
        self._continuous = False

    ########### ENV SPECIFIC PROPERTIES ###########
    @property
    def saturation(self):
        return self._saturation

    ########## GYM ENV INTERFACE ##################
    def step(self, action):
        commands = np.zeros(4)
        commands[1:] = action
        commands[0] = self.compute_thrust()
        commands = commands * self.normalize
        pwm = np.rint(np.dot(self.quadcopter.cmds_to_pwm, commands))
        pwm = np.minimum(pwm, self.quadcopter.saturation)
        pwm = np.maximum(pwm, 0)
        w = self.quadcopter.pwm_to_w(pwm)
        self.quadcopter.w.append(w)
        done = self.quadcopter.next_step(commands)

        self._query_duration -= self.quadcopter.dt_commands
        if self.continuous and not done and self.query_duration <= 0:
            self.random_query()

        self._state = self._f_compute_state(cmd=(commands / self.normalize))
        reward = self.f_reward(self._state[6:9])

        self.event_on_step(action=action, commands=commands, state=self.state, reward=reward, done=done)

        return self.state, reward, done, {}


    def set_continuous(self, quadcopter):
        self._quadcopter = quadcopter
        self.reset()
        self._continuous = True
        logging.debug(f"set_continuous(): continuous={self.continuous}")

    def set_episodic(self, quadcopter):
        self._quadcopter = quadcopter
        self.reset()
        self._continuous = False
        logging.debug(f"set_episodic(): continuous={self.continuous}")

    def _f_compute_state(self, cmd=np.zeros(4)):
        """ The Env Specific State Space Definition 
        """
        qs = np.zeros(10)
        if self.quadcopter is not None: qs = self.quadcopter.state
        return np.concatenate([
            qs[1:7],
            self.query - qs[7:],
            cmd,
            self.saturation
        ])

    def reset(self):
        """ Environment State Reset called by Gym Env 
        """
        self.quadcopter.w = []
        if self.quadcopter is not None:
            # Quadcopter State Reset
            self.quadcopter.reset(params=self.params)
        self._query = np.random.uniform(-0.4, 0.4, 3)
        self._query_active = True
        self._query_duration = np.random.uniform(0.1, 1.0)
        self._query_memory = {
            "rate_queries": [np.zeros(3), self.query, self.query],
            "t": [0.0, 0.0, self.query_duration]
        }
        self._saturation = np.ones(4)
        i = np.random.randint(0, 4)
        r = np.random.rand() / 5 + 0.8
        self._saturation[i] = r
        # This line was here before I took this, not sure if it can be useful at some point 
        # self.saturation[(i + 2) % 4] = r 

        if self.quadcopter is not None: self.quadcopter.set_saturation(self.saturation * self.normalize[0])

        self._state = self._f_compute_state()

        self._z_integral = 0.
        self._previous_z_error = 0.

        self.event_on_reset(additional_info=f"Stauration={self.saturation}")

        return self.state

    def random_query(self):
        """Random Query Generation 

        NOTES 
        - In this env it is still here but eventually it will be replaced with the external Query Generator 
        """
        if self.query_active:
            self._query_duration = np.random.uniform(0.1, 1.0)
            self._query = np.zeros(3)
        else:
            self._query_duration = np.random.uniform(0.1, 1.0)
            self._query = np.random.uniform(-0.6, 0.6, 3)
        self._query_memory['rate_queries'].append(np.copy(self.query))
        self._query_memory['rate_queries'].append(np.copy(self.query))
        self._query_memory['t'].append(self.quadcopter.t)
        self._query_memory['t'].append(self.quadcopter.t + self.query_duration)
        self._query_active = not self.query_active

    def get_reward(self, state):
        """Uses current pose of sim to return reward."""
        # return 1 + np.tanh(1 - np.abs(self.quadcopter.state[7:]).sum() / 6 * math.pi)
        # reward = 1. + max(-1., -np.abs(self.quadcopter.state[4:7]).sum() / math.pi)
        # return 1. - np.abs(self.quadcopter.state[4:7]).sum() / (math.pi)
        return max(-1., -np.abs(state[6:9]).sum() / (3 * math.pi))

    def render(self, mode='human', close=False):
        return

    def compute_thrust(self):
        z_error = self.target_z - self.quadcopter.z
        z_derivative, self._previous_z_error = (
            z_error - self.previous_z_error
        ) / self.quadcopter.dt, z_error
        self._z_integral += z_error * self.quadcopter.dt
        c = self.k_z[0] * z_error
        c += self.k_z[1] * self.z_integral
        c += self.k_z[2] * z_derivative + 48500
        c = max(0., min(c, 65535.)) / 65535.
        return c

    def close(self):
        return
