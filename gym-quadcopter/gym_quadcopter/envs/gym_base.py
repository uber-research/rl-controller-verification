import gym
import numpy as np 
from modules.query_generator.query_generator import RandomQueryGenerator, QueryArgs
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from utils.gen import TrainingParamsDict, sample
import math
import logging
import json
from pathlib import Path

class PIDBase:
    def __init__(self, name, dt_commands):
        self.name = name
        p = Path(__file__).parent.joinpath("config.json")
        with p.open() as config:
            config_json = json.load(config)
            self.pid_coeffs = config_json[self.name]
            self.normalize = config_json["normalize"]
        self.p = self.pid_coeffs["p"]
        self.i = self.pid_coeffs["i"]
        self.d = self.pid_coeffs["d"]
        self.dt_commands = dt_commands


class PIDThrust(PIDBase):
    def __init__(self, name, dt_commands):
        super().__init__(name, dt_commands)
        self.thrust_eq = self.pid_coeffs["thrust_eq"]
        self.previous_z_error = 0.0
        self.z_integral = 0.0
        self.set_is_pid_original()


    def set_is_pid_original(self):
        if self.name == "pid_thrust_original":
            self.is_pid_original = True
        else:
            self.is_pid_original = False

    def reset(self):
        self.previous_z_error = 0.0
        self.z_integral = 0.0

    def compute_thrust(self, target_z, current_z):
        if self.is_pid_original:
            return self.compute_thrust_original(target_z, current_z)
        else:
            return self.compute_thrust_main(target_z, current_z)

    def compute_thrust_main(self, target_z, current_z):
        z_error = target_z - current_z
        z_derivative = (z_error - self.previous_z_error) / self.dt_commands
        self.previous_z_error = z_error
        self.z_integral += z_error * self.dt_commands
        c = self.p * z_error
        c += self.i * self.z_integral
        c += self.d * z_derivative + self.thrust_eq
        c = max(0., min(c, self.normalize[0])) / self.normalize[0]
        return c

    def compute_thrust_original(self, target_z, current_z):
        z_error = target_z - current_z
        z_derivative = (z_error - self.previous_z_error) / self.dt_commands
        self.previous_z_error = z_error
        self.z_integral += (2 * z_error - z_derivative) * self.dt_commands
        c = self.p * (2 * z_error - z_derivative)
        c += self.i * self.z_integral + self.thrust_eq
        c = max(0., min(c, self.normalize[0])) / self.normalize[0]
        return c


class PIDRates(PIDBase):

    def __init__(self, name, dt_commands):
        super().__init__(name, dt_commands)
        self.rates_error = np.zeros(3)
        self.rates_integral = np.zeros(3)
        self.previous_rates_error = np.zeros(3)
        self.rates_derivative = np.zeros(3)

    def reset(self, resets):
        self.rates_integral *= resets
        self.rates_derivative *= resets

    def compute_PID(self, query, angular_rates):
        """Implemetaion of PID that controls the angular rates

        Given the coefficients of the PID (p, i, d), it computes the commands
        on the angular rates clipped with the normalization vector. It also uses the integral
        windup, meaning that the integral error of the PID will reset every time the error crosses
        zero.

        Returns:
            numpy arry: A vector of 3 floats corresponding to (cmd_phi, cmd_theta, cmd_psi).

    """
        rates_error = query - angular_rates
        # reset of the integral error at every zero crossing (windup)
        resets = [0 if rates_error[i]*self.previous_rates_error[i] <= 0
                  else 1 for i in range(3)]
        self.reset(resets)
        rates_derivative, self.previous_rates_error = (
            rates_error - self.previous_rates_error
        ) / self.dt_commands, rates_error
        self.rates_integral += rates_error * self.dt_commands
        c = self.p * rates_error
        c += self.i * self.rates_integral
        c += self.d * rates_derivative
        c = [max(-self.normalize[i],min(c[i-1],self.normalize[i]))/self.normalize[i] for i in range(1,4)]
        return c

class Rewards:
    """ Container Class for different Reward Functions

    NOTE
    - Now it just contains the standard dense reward consisting of tracking the query but it can be extended with more sophisticated rewards and eventually the STL based Rewards
    """
    @staticmethod
    def reward_1(state_query_delta):
        """ Triangle Reward 

        Default Reward: dense, instantaneous, depending on the difference between current state and query state

        Params
        - state_query_delta         :       np.array representing the difference between the current state and the query state

        NOTE
        - It was used with state_query_delta=env.state[6:9]

        QUESTIONS
        - Does it make sense there is no positive reward 
        """
        return max(-1., -np.abs(state_query_delta).sum() / (3 * math.pi))

    @staticmethod
    def reward_2(state_query_delta): 
        """ Old Reward 

        Params
        - state_query_delta         :       np.array representing the difference between the current state and the query state

        NOTE
        - It was used with state_query_delta=quadcopter.state[7:]
        """
        return 1 + np.tanh(1 - np.abs(state_query_delta).sum() / 6 * math.pi)

    @staticmethod
    def reward_3(state_query_delta):
        """ Old Reward 

        Params 
        - state_query_delta         :       np.array representing the difference between the current state and the query state

        NOTE
        - It was used with state_query_delta=quadcopter.state[4:7]
        """
        return 1. + max(-1., -np.abs(state_query_delta).sum() / math.pi)

    @staticmethod
    def reward_4(state_query_delta):
        """ Old Reward 

        Params 
        - state_query_delta         :       np.array representing the difference between the current state and the query state

        NOTE
        - It was used with state_query_delta=quadcopter.state[4:7]
        """
        return 1. - np.abs(state_query_delta).sum() / (math.pi)


class GymEnvBase(gym.Env):
    def __init__(self):
        pass

    @property
    def state(self): 
        return self._state

    @property
    def env_id(self): 
        return self._env_id

    @property
    def instance_index(self): 
        return self._instance_index

    @property
    def continuous(self):
        return self._continuous

    @property
    def quadcopter(self): 
        return self._quadcopter

    @property
    def iterations_since_last_reset(self):
        return self._iterations_since_last_reset

    @property
    def normalize(self): 
        return self._normalize

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def previous_z_error(self):
        return self._previous_z_error

    @property
    def target_z(self): 
        return self._target_z

    @property
    def z_integral(self):
        return self._z_integral

    @property 
    def query_active(self): 
        return self._query_active

    @property 
    def query_duration(self): 
        return self._query_duration

    @property
    def query(self): 
        return self._query 

    @property
    def query_memory(self): 
        return self._query_memory

    @property
    def query_gen(self):
        return self._query_gen

    @property
    def f_reward(self): 
        return self._f_reward

    @property
    def altitude_controller(self): 
        return self._altitude_controller

    @property
    def params(self): 
        return self._params

    def set_target_z(self, target_z):
        self._target_z = target_z
    
    def set_params(self, params):
        self._params = params

    @property
    def mask(self):
        return self._mask

    @property
    def box_states_high(self):
        return self._box_states_high

    @property
    def box_states_low(self):
        return self._box_states_low

    @property
    def commands(self):
        return self._commands

    def get_on_step_log(self):
        return self._on_step_log

    @property
    def pid_rates(self):
        return self._pid_rates

    @property
    def is_pid_rates(self):
        return self._is_pid_rates

    def __str__(self):
        return f"EnvID={self.env_id}, InstanceIndex={self.instance_index}, Mask={self.mask}, ObservationSpaceSize={self.observation_space.shape}"

    @staticmethod
    def make(env_id, instance_index, params, quadcopter=None, target_z=0.0, f_reward=Rewards.reward_1, query_class='something', query_classes={}, used_states = ['e_p']):
        """ This is a wrapper for the gym.make() factory method in the GymEnvBase class which allows to have a factory taking arguments

        It takes care of gym.make() and env config in the same place so it less error prone on the dev user side
        """
        env=gym.make(env_id)
        env.init(env_id=env_id, instance_index=instance_index, quadcopter=quadcopter, query_class=query_class, query_classes=query_classes, params=params, used_states = used_states)
        env.event_on_instantiation()
        return env

    def _init_state_log(self):
        self._on_step_log = "Reset"

    def init(self, env_id, instance_index, params, quadcopter=None, target_z=0.0, z_integral=0.0, previous_z_error=0.0, f_reward=Rewards.reward_1, query_class='something', query_classes={}, used_states = []):
        """ Custom Gym Env Constructor

        It is actually a pseudo-constructor which will be called contextually with the gym.make() factory method ensuring the Env Initialization is performed at construction time 
        """
        self._env_id = env_id
        self.set_instance_index(instance_index)
        self.set_params(params)
        self._iterations_since_last_reset = 0
        self._target_z = target_z
        self._z_integral = z_integral
        self._previous_z_error = previous_z_error
        self._quadcopter = quadcopter
        self._f_reward = f_reward
        self._continuous = False
        self._is_pid_rates = False
        if "pid_rates" in params and params["pid_rates"] != "None":
            self._is_pid_rates = True
        if self.is_pid_rates:
            self._pid_rates = PIDRates(params["pid_rates"], self.quadcopter.dt_commands)
        else:
            self._pid_rates = None
        self._commands = np.zeros(4)
        if "pid_thrust" in params and params["pid_thrust"] == "pid_thrust_original":
            self._altitude_controller = PIDThrust("pid_thrust_original", self.quadcopter.dt_commands)
        else:
            self._altitude_controller = PIDThrust("pid_thrust_main", self.quadcopter.dt_commands)
        self._normalize = self._altitude_controller.normalize




        # Flag saying if a query is already ongoing  
        self._query_active = False 

        # Duration of a Query 
        self._query_duration = None 

        # Actual Quert
        self._query = None 

        # Memory of Query 
        self._query_memory = None 

        self._params = params
        self._old_query = params["old_query"]
        if hasattr(self, 'set_query_classes') and hasattr(self, 'set_query_class'):
            self.set_query_classes(query_classes)
            self.set_query_class(query_class)

        self.all_states_string = [
            'u','v','w',
            'phi','theta','psi',
            'p','q','r',
            'q_p','q_q','q_r',
            'e_p','e_q','e_r',
            'thrust',
            'cmd_phi', 'cmd_theta', 'cmd_psi',
            'wg_x', 'wg_y', 'wg_z', 'wg_a',
            'saturation','pwm_1'
        ]
        self.check_used_states(used_states)
        self.is_pwm_1 = 'pwm_1' in used_states
        self._mask = np.isin(self.all_states_string, used_states)
        self._box_states_low=np.array([
            -30., - 30., - 30.,
            - math.pi, -math.pi, -math.pi,
            -5*np.pi, -5*np.pi, -5*np.pi,
            -5*np.pi, -5*np.pi, -5*np.pi,
            -5*np.pi, -5*np.pi, -5*np.pi,
            0., - 1., - 1., - 1.,
            -1., -1., -1., 0.,
            0.,0.
        ])
        self._box_states_high=np.array([
            30., 30., 30.,
            math.pi, math.pi, math.pi,
            5*np.pi, 5*np.pi, 5*np.pi,
            5*np.pi, 5*np.pi, 5*np.pi,
            5*np.pi, 5*np.pi, 5*np.pi,
            1., 1., 1., 1.,
            1, 1, 1., 1000,
            1., 65535.
        ])
        self._observation_space = gym.spaces.Box(
            low= self.box_states_low[self.mask],
            high= self.box_states_high[self.mask]
        )
        self._action_space = gym.spaces.Box(
            low=np.array([-1., -1., -1.]),
            high=np.array([1., 1., 1.]),
            )

        tp_desc = TrainingParamsDict(params)
        continuous = tp_desc.get_is_continuous()
        if continuous:
            if quadcopter is None: quadcopter=Quadcopter(T=tp_desc.qg_continuous.get_T_episode(), dt_commands=tp_desc.qg_continuous.get_dt_command(), dt=tp_desc.qg_continuous.get_dt())
            self.set_continuous(quadcopter=quadcopter)
        else:
            if quadcopter is None: quadcopter=Quadcopter(T=tp_desc.qg_episodic.get_T_episode(), dt_commands=tp_desc.qg_episodic.get_dt_command(), dt=tp_desc.qg_episodic.get_dt())
            self.set_episodic(quadcopter=quadcopter)

        self._init_state_log()


    def f_compute_commands(self, action): 
        """ Default Logic to compute commands 

        NOTE 
        - The Altitude is controlled by a non learnable altitude controller, by default it is a PID 
        - The Angular Rates are controlled by a learnable controller 
        """
        commands = np.zeros(4)
        commands[0] = self.altitude_controller.compute_thrust(target_z=self.target_z, current_z=self.quadcopter.z)
        if self.is_pid_rates:
            commands[1:] = self.pid_rates.compute_PID(self.query, self.quadcopter.state[7:])
        else:
            commands[1:] = action
        return commands 

    def f_state_2_str(self):
        temp = f"Quadcopter State = {self.quadcopter.state.shape}\n"
        temp += f"Quadcopter State[0]: TBD = {self.quadcopter.state[0]}\n"
        temp += f"Quadcopter State [1:7]: 6 DOF = {self.quadcopter.state[1:7]}\n"
        temp += f"Quadcopter State [7:]: 3 DOF targeted by query = {self.quadcopter.state[7:]}\n"
        temp += f"Query {self.query}\n"
        temp += f"Delta at t0 {self.quadcopter.state[7:] - self.query}"
        return temp

    def set_instance_index(self, instance_index):
        self._instance_index = instance_index
        logging.debug(f"Set Instance Index to {instance_index}")

    def get_env_id_full(self):
        return f"ENV ID={self.env_id}, Index={self.instance_index}"

    def event_on_instantiation(self):
        logging.info(f"Created an Instance of Environment ID={self.env_id} created, continuous={self.continuous}")

    def event_on_reset(self, additional_info=""):
        logging.debug(f"({self.env_id}) EVENT: Reset()\n State={self.state}\n Query=(query_target={self.query}, query_duration={self.query_duration}, query_active={self.query_active})\n Additional_Info={additional_info} END\n -------------")

    def event_on_step(self, action, commands, state, reward, done, additional_info=""):
        self._on_step_log = f"({self.env_id}) EVENT: Step(), Iteration_since_last_reset={self.iterations_since_last_reset}, action={action}, normalized_commands={commands}, state={state}, reward={reward}, done={done}, Additional_Info={additional_info} END\n ------------"
        logging.debug(self._on_step_log)

    def check_used_states(self, used_states):
        for state in used_states:
            if state not in self.all_states_string:
                raise RuntimeError(f'the state {state} is not in all_states_string')
