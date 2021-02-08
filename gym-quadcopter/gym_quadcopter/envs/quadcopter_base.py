import numpy as np
import math

class Quadcopter_Base: 
    """ Base Class for Quadcopter 
    It contains all the default behaviors 
    """
    def __init__(self):
        """ Base Class of Standard Quadcopter
        """
        self._init_state = np.zeros(10)
        self._init_saturation = np.ones(4) * 65535

    @staticmethod
    def build_state(z=0.0, v_x=0.0, v_y=0.0, v_z=0.0, yaw=0.0, pitch=0.0, roll=0.0, yaw_rate=0.0, pitch_rate=0.0, roll_rate=0.0):
        return np.array([z, v_x, v_y, v_z, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate])

    @property
    def z(self):
        return self._state[0]

    @property
    def velocity(self):
        return self._state[1:4]

    @property
    def velocity_x(self):
        return self._state[1]

    @property
    def velocity_y(self):
        return self._state[2]

    @property
    def velocity_z(self):
        return self._state[3]

    @property
    def abs_roll(self):
        return self._state[4]

    @property
    def abs_pitch(self):
        return self._state[5]

    @property
    def abs_yaw(self):
        return self._state[6]

    @property
    def rate_roll(self):
        return self._state[7]

    @property
    def rate_pitch(self):
        return self._state[8]

    @property
    def rate_yaw(self):
        return self._state[9]

    def set_z(self, z=None):
        if z is not None: self._state[0] = z

    def set_velocity(self, v_x=None, v_y=None, v_z=None):
        if v_x is not None: self._state[1] = v_x
        if v_y is not None: self._state[2] = v_y
        if v_z is not None: self._state[3] = v_z

    def set_abs_angles(self, abs_roll=None, abs_pitch=None, abs_yaw=None): 
        if abs_roll is not None: self._state[4] = abs_roll
        if abs_pitch is not None: self._state[5] = abs_pitch
        if abs_yaw is not None: self._state[6] = abs_yaw

    def set_rate_angles(self, rate_roll=None, rate_pitch=None, rate_yaw=None): 
        if rate_roll is not None: self._state[7] = rate_roll
        if rate_pitch is not None: self._state[8] = rate_pitch
        if rate_yaw is not None: self._state[9] = rate_yaw

    def _z_init(self):
        self.set_z(z=0.0)

    def _abs_lin_vel_init(self):
        self.set_velocity(v_x=0.0, v_y=0.0, v_z=0.0)

    def _abs_angle_init(self):
        self.set_abs_angles(abs_roll=0, abs_pitch=0, abs_yaw=0)

    def _angular_rates_init(self):
        self.set_rate_angles(rate_roll=0.0, rate_pitch=0.0, rate_yaw=0.0)

    def _default_reset_state(self): 
        self._z_init()
        self._abs_lin_vel_init()
        self._abs_angle_init()
        self._angular_rates_init()
