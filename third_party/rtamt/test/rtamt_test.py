##
# Advanced Technology Center - Paris - 2020
# Uber France Software & Development
##
"""
Provides unit tests for rtamt module
"""

import unittest
import rtamt

class TestRtamt(unittest.TestCase):
    """ This class regroups the unit tests
    """

    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_monitor_offline(self):
        # From https://github.com/nickovic/rtamt/blob/master/examples/basic/monitor_offline.py
        dataSet = {
             'a': [(0, 100.0), (1, -1.0), (2, -2.0)],
             'b': [(0, 20.0), (1, 2.0), (2, -10.0)]
        }

        spec = rtamt.STLSpecification(1)
        spec.name = 'HandMadeMonitor'
        spec.declare_var('a', 'float')
        spec.declare_var('b', 'float')
        spec.declare_var('c', 'float')
        spec.spec = 'c = always(a<=2 and b>=1)'
        spec.parse()
        robustness = spec.offline(dataSet)
        self.assertEqual(robustness, -98.0)


if __name__ == '__main__':
    unittest.main()
