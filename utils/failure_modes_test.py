##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'gym_gen.py'
"""

import unittest
from gym_quadcopter.envs.failure_modes import Saturation, Filter
import numpy as np


class TestGymGen(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_saturation_1(self):
        sat = Filter(
            is_active=True,
            func=Saturation(
                    sat_min=np.array([0.5, 0.5, 0.5, 0.5]),
                    sat_max=np.array([3.5, 3.5, 3.5, 3.5]) 
                )
            )
        #sat.is_active = False
        self.assertTrue(np.all( sat(np.ones(4) * 5) == np.ones(4)*3.5 ))
        self.assertTrue(np.all( sat(np.zeros(4)) == np.ones(4)*0.5 ))
        self.assertTrue(np.all( sat(np.array([0.0, 5.0, 6.0, 0.3])) == np.array([0.5, 3.5, 3.5, 0.5])))

if __name__ == '__main__':
    unittest.main()
