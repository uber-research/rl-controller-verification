# Library providing the Piecewise Constant Function representation used to represent signals, implemented with Numba acceleration 
import numpy as np
from numpy import inf
from numba import float64 
from numba.experimental import jitclass

spec = [
    ('xs', float64[:]),          # an array field
    ('ys', float64[:]),          # an array field
    ('yd', float64),             # a simple scalar field
    ('min_x', float64),
    ('max_x', float64),
]

@jitclass(spec)
class PwcSignal:
    """
    A callable piecewise constant signal.
    """

    def __init__(self, xs, ys, yd):
        """
        A piecewise constant function defined as follows:
        (-inf, x0)    -> yd
        [x_i, x_i+1)  -> y_i  for 0 <= i < n-1
        [x_n-1, +inf) -> y_n-1
        :param xs: sorted numpy array of floats
        :param ys: numpy array of floats
        :param yd: float
        """
        assert len(xs) == len(ys)
        self.xs = xs
        self.ys = ys
        self.yd = yd
        self.min_x = inf
        self.max_x = -inf

    def apply(self, x):
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        i = np.searchsorted(self.xs, x, side='right') - 1
        if i == -1:
            return self.yd
        else:
            return self.ys[i]
        
    def __call__(self, x):
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        i = np.searchsorted(self.xs, x, side='right') - 1
        if i == -1:
            return self.yd
        else:
            return self.ys[i]

    def idxrange(self, x_min, x_max):
        """
         Returns generator
             (i, ..., j)
         where:
         - xs[i-1] < x_min <= xs[i]
         -   xs[j] <= x_max < xs[j+1]
         :param x_min: num
         :param x_max: num
         :return: generator
        """
        assert x_min <= x_max
        i = np.searchsorted(self.xs, x_min, side='left')
        j = np.searchsorted(self.xs, x_max, side='right')
        return range(i, j)
   
    def xsrange(self, x_min, x_max):
        """
         Returns generator
             (xs[i], ..., xs[j])
         where:
         - xs[i-1] <  x_min <= xs[i]
         -   xs[j] <= x_max < xs[j+1]
         :param x_min: num
         :param x_max: num
         :return: generator
         """
        assert x_min <= x_max
        i = np.searchsorted(self.xs, x_min, side='left')
        j = np.searchsorted(self.xs, x_max, side='right')
        return (self.xs[i] for i in range(i, j))

    def reset(self):
        """
        Resets the out of bounds access trackers.
        """
        self.min_x =  inf
        self.max_x = -inf

    def oob(self):
        """
        True iff the signal was accessed outside of the bounds of first and last breakpoints.
        """
        return self.min_x < self.xs[0] or self.max_x > self.xs[-1]
