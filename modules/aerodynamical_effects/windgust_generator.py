from math import sin, cos, tan, pi
import numpy as np
import json
from dataclasses import dataclass
from typing import Tuple


@dataclass
class WindgustGenerator:
    """Creates a windgust corresponding to the parameters.

    This module is used to create a randomly generated windgust that alternates between two modes:
    'rest' in which the amplitude is set to 0
    'gust' in which the amplitude follows a specefic model detailled in the method sample

    Attributes:
        h_range (tuple): The range of the duration of the gust
        rest_range (tuple): The range of the duration of the rest mode
        agmax (float): The amplitude maximum that the gust can reach

    """


    h_range: Tuple[float, float]
    rest_range: Tuple[float, float]
    agmax: float

    def sample(self, T):
        """Creates a windgust corresponding to the parameters.

        This method generates a callable: 'value' that will compute the value of
        the windgust for whatever value of t. The windgust generated corresponds to a model where
        there are two modes: rest, in which the value of the windgust is always equal to 0
        and gust, in which the value of the windgust corresponds to the following formula chosen in our paper:
        (ag/2)*(1-cos(pi*(t-t_start_gust)/duree_windgust))*wind_direction.
        The mode 'gust' is activated by the boolean current_is_gust once in two.
        the magnitude, the duation of the mode gust and the duration of the mode rest are randomly chosen
        inside the given intervals in attribute: h_range for the gust duration, rest_range for the rest duration and agmax for
        the maximum amplitude.
        sample computes the breakpoints to give to the method 'value' which will be called at every step to return the value
        of the windgust (vector in 3D) in the inertial referential.


        Args:
            T(int): The length of the full episod
        """
        t = 0
        list_times = [0] # A list of the times when the mode changes
        list_vdir = [] # A list of arrays where each array corresponds to the direction of the wind for a breakpoint
        list_ag = [] # A list of the maximum amplitude reached for every breakpoint
        current_is_gust = False
        list_is_gust = [current_is_gust]
        while t < T:
            v_dir = np.random.uniform(-1, 1, 3)
            v_dir /= np.linalg.norm(v_dir)
            ag = 0
            if current_is_gust:
                ag = np.random.uniform(0, self.agmax)
                h = np.random.uniform(self.h_range[0], self.h_range[1])
                t += h
                list_times.append(t)
            else:
                rest = np.random.uniform(
                    self.rest_range[0], self.rest_range[1])
                t += rest
                list_times.append(t)
            list_ag.append(ag)
            list_vdir.append(v_dir)
            current_is_gust = not current_is_gust # Changes the mode
            list_is_gust.append(current_is_gust)
        list_times = np.array(list_times)

        def value(t):
            assert(t >= list_times[0])
            i = list_times.searchsorted(t, side='right') - 1 # Searches for the correct breakpoint
            if list_is_gust[i]:
                return (list_ag[i] / 2) * (1 - cos(2 * pi * (t - list_times[i]) / \
                    (list_times[i+1] - list_times[i]))) * list_vdir[i]
            else:
                return np.zeros(3)
        return value
