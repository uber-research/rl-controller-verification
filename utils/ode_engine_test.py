##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'ode_engine.py'
"""

import unittest
import numpy as np
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.ode_engine import rKN


class TestODEEngine(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_dynamic_temporal_consistency_time_scales_1(self):
        q = Quadcopter()
        n = 10
        current_state = np.zeros(10)
        w = np.ones(4)
        fm = np.dot(q.w2_to_fm, w**2)

        for dt in [0.03, 0.0001, 1e-7]:
            obs_trace = np.zeros([10,10])
            exp_trace = np.zeros([10,10])
            state = np.zeros(n)
            for i in range(10):
                state = rKN(state, q.ODE(fm), n, dt)
                exp_trace[i] = state

            state = np.zeros(n)
            for i in range(10):
                state = rKN(state, q.ODE(fm), n, dt)
                obs_trace[i] = state

            self.assertTrue(np.all( obs_trace == exp_trace), msg=f"Obs_Trace={obs_trace}\nExp_Trace={exp_trace}")


if __name__ == '__main__':
    unittest.main()
