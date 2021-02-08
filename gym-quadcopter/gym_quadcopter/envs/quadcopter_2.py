import logging
import json
from math import sin, cos, tan, pi
import os.path
import numpy as np
import pkg_resources
from gym_quadcopter.envs.quadcopter_base import Quadcopter_Base
from utils.gen import sample
from gym_quadcopter.envs.ode_engine import rKN
from gym_quadcopter.envs.episode import Episode
from pathlib import Path
from typing import Callable

class Quadcopter(Quadcopter_Base):
    """Quadcopter Dynamical Model used in envs.

    Attributes:
        init_state  : The initial state of the quadcopter (default: np.zeros(10))
        dt          : Time Granularity of the Simulation (default: 0.01)
        dt_commands : Time Granularity of the Commands (default: 0.03)
        T           : Episode Duration (default: 5.0)

    Internal State: 
        t           : [float]               Internal Physical Time Counter
        state       : [np.array(10)]        The Quadcopter State 


    Quadcopter State Description 
    0               : Absolute Z  
    1,2,3           : Linear Velocities 
    4,5,6           : Absolute Angles 
    7,8,9           : Angular Rates 
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            init_state=np.zeros(10),
            init_saturation=np.ones(4) * 65535,
            dt=0.01,
            dt_commands=0.03,
            T=5.,
            saturation_motor_min = 1.,
            windgust_generator = None,
            aero = False,
            cb_step : Callable = None
    ):
        """ Quadcopter Constructor 

        Args:
            cb_step         (callback)          :           Callback called at every ODE integration step

            TODO:Add description for all the other fields
        """
        super().__init__()
        p = Path(__file__).parent.joinpath("config.json")
        with p.open() as config:
            self.params = json.load(config)["quadcopter"]
        self.c1 = self.params["C1"]
        self.c2 = self.params["C2"]
        self.ct = self.params["CT"]
        self.cd = self.params["CD"]
        self.d = self.params["d"]
        self.h = self.params["h"]
        self.ix = self.params["Ix"]
        self.iy = self.params["Iy"]
        self.iz = self.params["Iz"]
        self.m = self.params["m"]
        self.g = self.params["g"]
        self.k = self.params["K"]
        self._saturation_motor_min = saturation_motor_min
        self._init_state = init_state
        self._init_saturation = np.copy(init_saturation)
        self.w = []
        self._saturation = np.copy(self._init_saturation)
        self.aero = aero
        self.windgust_generator = windgust_generator
        self.windgust_sample = None
        self._state = np.copy(self._init_state) # State changes so here it is necessary to copy it  
        self._dt = dt
        self._dt_commands = dt_commands
        self._t = 0.0
        self._T = T
        self.n = self.state.shape[0]
        self.cb_step = cb_step
        self.init_matrices()
        self._commands = np.zeros(4)
        self._command_factor = int(self.dt_commands / self.dt)
        self.episode = Episode(T=T, altitude_min=-1000.0)
        self.is_training = True
        self.pwm_1 = 50000/self.saturation[1]
        self.saturated = 0
        self.not_saturated = 0
        logging.info(f"[QUADCOPTER]: Created an instance of Quadcopter with T={self.T}, dt={self.dt}, dt_commands={self.dt_commands}")

    @property
    def saturation(self):
        return self._saturation

    @property
    def init_saturation(self):
        return self._init_saturation

    @property
    def saturation_motor_min(self):
        return self._saturation_motor_min

    @property
    def saturation_motor(self):
        return self._saturation_motor

    @property
    def dt(self): 
        return self._dt

    @property 
    def init_state(self): 
        return self._init_state

    @property
    def state(self): 
        return self._state

    @property
    def t(self): 
        return self._t

    @property 
    def T(self): 
        return self._T 

    @property
    def dt_commands(self): 
        return self._dt_commands

    @property
    def commands(self): 
        return self._commands

    @property
    def command_factor(self): 
        return self._command_factor

    @property
    def windgust(self):
        return self._windgust

    def reset(self, params=None, state=None, saturation=None):
        """ Quadcopter State Reset Function
        """
        super()._default_reset_state()
        self._windgust = np.zeros(4)
        self._saturation_motor = self.saturation_motor_min
        if params is not None and 'quadcopter' in params:
            super()._default_reset_state()
            # Reset Policy is currently considered an optional section 
            # If it is not present, the default reset policy is used 
            if 'reset_policy' in params['quadcopter']:
                base = params['quadcopter']['reset_policy']
                # Overriding some parts of the Quadcopter Reset
                self.set_z(z=sample(yaml_config=base['abs_z']))
                self.set_velocity(v_x=sample(yaml_config=base['velocity_x']))
                self.set_velocity(v_y=sample(yaml_config=base['velocity_y']))
                self.set_velocity(v_z=sample(yaml_config=base['velocity_z']))
                self.set_abs_angles(abs_roll=sample(yaml_config=base['abs_roll']))
                self.set_abs_angles(abs_pitch=sample(yaml_config=base['abs_pitch']))
                self.set_abs_angles(abs_yaw=sample(yaml_config=base['abs_yaw']))
                self.set_rate_angles(rate_roll=sample(yaml_config=base['rate_roll']))
                self.set_rate_angles(rate_pitch=sample(yaml_config=base['rate_pitch']))
                self.set_rate_angles(rate_yaw=sample(yaml_config=base['rate_yaw']))
            if self.windgust_generator is not None:
                self.windgust_sample = self.windgust_generator.sample(self.T)
        elif state is not None: 
            self._state = state 
        self._saturation[0] = self.saturation_motor * self.init_saturation[0]
        self._t = 0.0


    def next_step(self, commands, pwm=False, impl_version=0):
        """Evolution Function - External API
        
        Parameters
        ----------
        commands : numpy.ndarray (4,)
            Control Signal 
        pwm : bool
            Control with cmd or pwm directly 
        """

        def _next_step(self, commands, pwm=False):
            """Evolution Function - Original Implementation 

            Parameters
            ----------
            commands : numpy.ndarray (4,)
                Control Signal 
            pwm : bool
                Control with cmd or pwm directly 
            """
            self._commands = commands 
            for _ in range(self.command_factor):        # The same commands get repeated 
                # FIXME: This hardcoded '10' needs to be transformed into a param <FIXME_t_20191121_1432_1>

                # If it is defined, call the ODE Integration Callback passing time and state
                if self.cb_step is not None:
                    self.cb_step()

                self.step_RKn(10, pwm)
                self._t += self.dt

                if self.check_stop():
                    break
            return self.check_stop()

        def _saturate_and_get_w(x): 
            x = np.minimum(x, self.saturation)
            x = np.maximum(x, 0)
            return self.pwm_to_w(x)

        def _get_pwm(commands, pwm_mode=False): 
            if not pwm_mode:                                                            # If input is not pwm then a conversion in pwm is needed  
                return np.rint(np.dot(self.cmds_to_pwm, commands))
            else:
                return commands

        def _new_next_step(self, commands, pwm_mode=False): 
            for _ in range(self.command_factor):                                        # The same commands get repeated 
                pwm = _get_pwm(commands, pwm_mode)
                w = _saturate_and_get_w(pwm)
                fm = np.dot(self.w2_to_fm, w**2)
                self._state = rKN(x=self.state, fx=self.ODE(fm), n=self.n, hs=self.dt)
                self._t += self.dt
                if self.check_stop():
                    break
            return self.check_stop()
            
        if(impl_version==0):                                        # impl_version = 0 means original implementation  
            return _next_step(self, commands, pwm)                  # Original Implementation 
        else: 
            return _new_next_step(self, commands, pwm)              # New Implementation 

    def step_RKn(self, n, pwm_mode=False):
        if not pwm_mode:                                                    # Out of this, `pwm` will contain the actual 
            pwm = np.rint(np.dot(self.cmds_to_pwm, self.commands))          #
        else:                                                               # 
            pwm = self.commands                                             # 
        pwm = np.minimum(pwm, self.saturation)
        pwm = np.maximum(pwm, 0)
        if pwm[0] == (0.8 * self.saturation[1]):
            self.saturated += 1
        else:
            self.not_saturated += 1
        w = self.pwm_to_w(pwm)
        fm = np.dot(self.w2_to_fm, w**2)
        windspeed = np.zeros(3)
        f_aero = [0]*6
        if self.aero == True:
            if self.windgust_sample is not None:
                windspeed = self.windgust_sample(self.t)
            f_aero = self.compute_forces(pwm, windspeed)
        norm_windgust = np.linalg.norm(windspeed)
        if norm_windgust > 0:
            for i in range(len(windspeed)):
                self._windgust[i] = windspeed[i] / norm_windgust
            self._windgust[-1] = norm_windgust
        else:
            self._windgust = np.zeros(4)
        self._state = rKN(x=self.state, fx=self.ODE(fm, f_aero), n=n, hs=self.dt)

    def init_matrices(self):
        """Conversion Matrix 4x4 from Commands to pwm"""
        self.cmds_to_pwm = np.array([
            [1., -0.5, -0.5, -1.],
            [1., -0.5, 0.5, 1.],
            [1., 0.5, 0.5, -1.],
            [1., 0.5, -0.5, 1.]
        ])

        self.w2_to_fm = np.array([
            self.ct * np.ones(4),
            self.d * self.ct / np.sqrt(2) * np.array([-1., -1., 1., 1.]),  # noqa E261
            self.d * self.ct / np.sqrt(2) * np.array([-1., 1., 1., -1.]),  # noqa E261
            self.cd * np.array([-1., 1., -1., 1.])
        ])
        self.pwm_to_w = lambda x: self.c1 * x + self.c2

    def set_saturation(self, sat):
        self._saturation = sat

    def ODE(self, fm, f_aero=[0]*6):
        def f(t, x):
            ode = np.empty(10)
            ode[0] = -np.sin(x[5]) * x[1] + np.cos(x[5]) * np.sin(x[4]) * x[2]
            ode[0] += np.cos(x[5]) * np.cos(x[4]) * x[3]

            ode[1] = x[9] * x[2] - x[8] * x[3]
            ode[1] += np.sin(x[5]) * self.g + f_aero[0] / self.m

            ode[2] = -x[9] * x[1] + x[7] * x[3] - np.cos(
                x[5]) * np.sin(x[4]) * self.g + f_aero[1] / self.m

            ode[3] = x[8] * x[1] - x[7] * x[2] - np.cos(
                x[5]) * np.cos(x[4]) * self.g + fm[0] / self.m + f_aero[2] / self.m  # noqa E261
            ode[4] = x[7] + np.cos(x[4]) * np.tan(
                x[5]) * x[9] + np.tan(x[5]) * np.sin(x[4]) * x[8]
            ode[5] = np.cos(x[4]) * x[8] - np.sin(x[4]) * x[9]
            ode[6] = x[9] * np.cos(x[4]) / np.cos(
                x[5]) + x[8] * np.sin(x[4]) / np.cos(x[5])
            ode[7] = (
                self.iy - self.iz
            ) / self.ix * x[8] * x[9] + 1. / self.ix * fm[1] + f_aero[3]  # noqa E261
            ode[8] = (
                self.iz - self.ix
            ) / self.iy * x[7] * x[9] + 1. / self.iy * fm[2] + f_aero[4]  # noqa E261 Corrected EG may 2020 (x8*x9->x7*x9)
            ode[9] = (
                self.ix - self.iy
            ) / self.iz * x[7] * x[8] + 1. / self.iz * fm[3] + f_aero[5]  # noqa E261 Corrected EG may 2020 (x8*x9->x7*x8)
            return ode
        return f

    def check_stop(self):
        return self.episode.is_end(t=self.t, quadcopter=self)
        # return self.state[0] < -5. or self.t > self.T

    def R(self):
        """ Changing base matrix
        R() returns the changing base matrix from
        the inertial referential to the quadcopter referential
        """
        phi, theta, psi = self.state[4:7]
        return np.array([
                        [cos(psi)*cos(theta), cos(psi)*sin(theta)*sin(phi)-cos(phi)*sin(psi), sin(psi)*sin(phi)+cos(psi)*cos(phi)*sin(theta)],
                        [cos(theta)*sin(psi), cos(psi)*cos(phi)+sin(psi)*sin(theta)*sin(phi), cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi)],
                        [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)]])

    def R_T(self):
        """ Changing base matrix
        R_T() returns the changing base matrix from
        the quadcopter referential to the inertial referential
        """
        return np.transpose(self.R())

    def compute_forces(self, pwm, wa):
        forces_body = np.zeros(3)
        moments = np.zeros(3)
        R = self.R()
        R_T = self.R_T()
        _, u, v, w, _, _, _, p, q, r = self._state
        for i in range(4):
            omega = self.c1*pwm[i] + self.c2
            x_rotor = sin((pi/2)*i + 3*pi/4)
            y_rotor = cos((pi/2)*i + 3*pi/4)
            rotors = [self.d*x_rotor,
                     self.d*y_rotor, self.h]
            linear_velocity_rotors = np.cross([p, q, r], rotors) + [u, v, w]
            wr = linear_velocity_rotors - np.dot(R_T, wa)
            force_body = omega*(np.dot(self.k, wr))
            forces_body += force_body
            moments += np.cross(rotors, force_body)
        return np.append(forces_body, moments)
