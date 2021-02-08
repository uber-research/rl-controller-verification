##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'ode_engine.py'
"""

import unittest
import numpy as np
from gym_quadcopter.envs.episode import Episode
from gym_quadcopter.envs.quadcopter_2 import Quadcopter

class TestEpisodeTest(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_episode_1(self):
        q = Quadcopter()
        q.set_z(-20.0)
        t = 1.0
        episode = Episode(T=5.0, altitude_min=-10.0, altitude_max=None)
        self.assertTrue(episode.is_end(t=t, quadcopter=q))

if __name__ == '__main__':
    unittest.main()
