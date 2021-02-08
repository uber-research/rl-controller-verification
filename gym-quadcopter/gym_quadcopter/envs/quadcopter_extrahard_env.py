import gym
from gym import error, spaces, utils
from gym.utils import seeding
from model.models import Quadcopter

class QuadcopterExtraHardEnv(gym.ExtraHardEnv):

    def __init__(self):
        self.x = 0

    def step(self, action):
        return

    def reset(self):
        return

    def render(self, mode='human', close=False):
        return

    def close():
        return
