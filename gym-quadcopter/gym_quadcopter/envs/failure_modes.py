import numpy as np
from typing import NamedTuple

class Saturation(NamedTuple):
    sat_min: np.ndarray = np.zeros(4)
    sat_max: np.ndarray = np.ones(4)

    def __call__(self, x):
        return np.maximum(np.minimum(x, self.sat_max), self.sat_min)



class Filter:
    def __init__(self, is_active, func=Saturation()):
        self._is_active = is_active
        self._func = func

    @property
    def is_active(self):
        return self._is_active

    def set_is_active(self, is_active: bool):
        if not isinstance(is_active, bool):
            raise TypeError('is_active must be bool')
        self._is_active = is_active

    @property
    def func(self):
        return self._func

    def set_is_active(self, is_active):
        self._is_active = is_active

    def __call__(self, x):
        if self.is_active:
            return self.func(x)
        else:
            return x

