import logging
import gym
import math
import numpy as np
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import GymEnvBase
from modules.query_generator.smooth_query_generator import QueryClass

class QuadcopterEnv5(GymEnvBase):
    """Gym Env

    - The Env Registration ID identifies only an ID type
    - The Full Env ID is defined by the Env Registration ID and the Instance Index

    Observation Space Dim     : 14
    - Speed in 3D             : 03
    - Absolute Angles         : 03
    - Angular Rates - Queries : 03
    - Altitude                : 01
    - z_integral              : 01
    - Cmd                     : 03

    Attributes:
        env_id          : Env Registration ID
        instance_index  : The Index of the Instance in the Vectorized Env Data Structure
        quadcopter      : The Quadcopter Model used for interacting with the env

"""

    metadata = {'render.modes': ['human']}

    def __init__(self, pid_gains=[3000., 300., 2000.]):
        super().__init__()
        self._k_z = [3000., 300., 2000.]
        # Now Quadcopter gets passed from the external
        #self.init(env_id=5, instance_index=0, quadcopter=None, continuous=False)
        self._observation_space = gym.spaces.Box(
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
        self._action_space = gym.spaces.Box(
            low=np.array([-1., -1., -1.]),
            high=np.array([1., 1., 1.]),
        )
        np.set_printoptions(precision=3)

    @property
    def saturation(self):
        return self._saturation

    @property
    def query_classes(self):
        return self._query_classes

    @property
    def query_class(self):
        return self._query_class

    def set_query_classes(self,query_classes):
        self._query_classes = QueryClass.make_from(query_classes)

    def set_query_class(self,query_class):
        self._query_class = query_class

    def step(self, action):
        commands = np.zeros(4)
        commands[1:] = action
        commands[0] = self.compute_thrust()
        commands = commands * self.normalize
        pwm = np.rint(np.dot(self.quadcopter.cmds_to_pwm, commands))
        pwm = np.minimum(pwm, self.quadcopter.saturation)
        pwm = np.maximum(pwm, 0)
        w = self.quadcopter.pwm_to_w(pwm)
        done = self.quadcopter.next_step(commands)
        self.random_query()
        self._state = self._f_compute_state(cmd=commands/self.normalize)
        reward = self.f_reward(self._state[6:9])
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
            self.quadcopter.state[1:7],
            self.query - self.quadcopter.state[7:],
            [self.quadcopter.state[0], self._z_integral],
            np.zeros(3)
        ])

    def reset(self):
        """ Environment State Reset called by Gym Env
        """
        self.quadcopter.w = []
        if self.quadcopter is not None:
            # Quadcopter State Reset
            self.quadcopter.reset(params=self.params)
        self._query_memory = {
            "rate_queries": [np.zeros(3)],
            "t": [0.0]
        }

        self._query_sample = None
        if self.query_class is self._query_classes.keys():
            self._query_sample =self._query_classes[self.query_class].sample(self.quadcopter.T,self.quadcopter.dt_commands)
        self.random_query()
        self._state = self._f_compute_state()
        self._z_integral = 0.
        self._previous_z_error = 0.
        return self.state

    def random_query(self):
        self._query = np.random.uniform(-0.4, 0.4, 3)
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

    def compute_thrust(self):
        z_error = self.target_z - self.quadcopter.state[0]
        z_derivative, self._previous_z_error = (
            z_error - self.previous_z_error
        ) / self.quadcopter.dt, z_error
        self._z_integral += z_error * self.quadcopter.dt
        c = (
            self.k_z[0] * z_error + self.k_z[1] * self.z_integral
        ) + self.k_z[2] * z_derivative + 48500
        c = max(0., min(c, 65535.)) / 65535.
        return c

    def close(self):
        return
