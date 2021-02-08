"""
Provides integration test for file 'perf_observer.py'
"""

import unittest
from perf_observer import load_signals, PerfObserver

class TestPropertiesObserver(unittest.TestCase):
    """ This class regroups the unit tests
    """
        
    def _get_signals(self, basename='properties_observer/data/trajectory_cp_6_iteration_0_', tlimit=5.0):
        """ Loads and returns example signals 

        Args:
            tlimit  :   Time limit for the loaded trace
                        WHY: The trace can be very long so it is possible to just load a subset of it

        Returns: 
            t   :   Time Signal
            q   :   Query Signal
            x   :   NN Output
        """
        signals = load_signals(basename=basename, t_is_hex=False, tlimit=tlimit)
        t = signals['t']
        x = signals['p']
        q = signals['query_p']
        return t,x,q

    def _get_performance_observer(self):
        """ Instantiates the default performance observer

        Returns:
            PerfObserver    :       The default performance observer
        """
        return PerfObserver(max_sig_len=10000, prop_params=PerfObserver.default_prop_params())

    def setUp(self):
        """ Setting up the test loading signals and preparing the observer
        """
        self._signals = self._get_signals()
        self._po = self._get_performance_observer()


    def test_imports(self):
        """ This test does nothing except cheking that import works
        """
        pass

    def test_integration_on_concrete_trace(self):
        """ Shows how to use the performance observer on a concrete trace 
        """
        t,x,q = self._signals
        self._po.observe(t=t, x=x, q=q, save_eval=False)
        print(f"Observation Results\n{self._po}")

if __name__ == '__main__':
    unittest.main()
