##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for file 'eval_model.py'
"""


import numpy as np
import unittest
import model.eval_model


class TestEvalModel(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_mean_confidence_interval_1(self):
        temp = [{'in': np.array([1.,2.,3.]), 'out': np.array([2.0, 2.484])}]
        f = model.eval_model.mean_confidence_interval
        for e in temp:
            self.assertTrue(np.allclose(f(e['in']), e['out'], rtol=1e-3), msg=f"{f(e['in'])} not close to {e['out']}")


if __name__ == '__main__':
    unittest.main()
