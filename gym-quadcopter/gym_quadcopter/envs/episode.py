import numpy as np

class Episode:
    """ Contains general params about the Episode

    We set the altitude_min to -1000 in order to see the full trace,
    we only focus on the attitude controller and let the altitude be
    very low.

    Args:
        T (float): The maximum duration of the episod
        altitude_min (float) : A criteria to stop the episod.
            Default to -1000.0 since we are not interested in
            stopping the episod because of the altitude for the
            moment.
        altitude_max (float) : Maximum altitude reached before
            the episod stops. Default to None for the same reason mentionned
            above.
    """

    def __init__(self, T=5.0, altitude_min=-1000.0, altitude_max=None):
        self._T = T
        self._altitude_min = altitude_min
        self._altitude_max = altitude_max

    @property
    def T(self):
        return self._T

    @property
    def altitude_min(self):
        return self._altitude_min
    
    @property
    def altitude_max(self):
        return self._altitude_max

    def is_end(self, quadcopter, t):
        if t > self.T:
            return True
        if (self.altitude_min is not None and quadcopter.z < self.altitude_min) or (self.altitude_max is not None and quadcopter.z > self.altitude_max):
            return True
        return False


