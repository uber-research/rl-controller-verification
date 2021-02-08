import numpy as np 
import logging
import gym
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from utils.gen import TrainingParamsDict
from gym_quadcopter.envs.quadcopter_2 import Quadcopter
from gym_quadcopter.envs.gym_base import GymEnvBase
from modules.aerodynamical_effects.windgust_generator import WindgustGenerator
#from stable_baselines.common.vec_env import DummyVecEnv

def f_fwgym_get_env(env_id, used_states, instance_index, query_classes, query_class, params):
    """ Function that instantiates the Gym Env 

    ---
    Params 

    env_id              : (str)     Env ID
    instance_index      : (int)     Instance Index used in case of multiple environments
    query_classes       : (TBD)     Query Classes 
    query_class         : (TBD)     Query Class
    params              : (Dict)    Training or Testing Params defined in the related config file in a structure, some of them have an effect on the env internal details
    """
    tp_desc = TrainingParamsDict(params)
    aero = params['aero']['enabled']
    rest_range = params['aero']['windgust']["rest_range"]
    period_range = params['aero']['windgust']['h_range']
    magnitude_max = params['aero']['windgust']['magnitude_max']
    windgust_generator = WindgustGenerator(h_range=period_range, rest_range=rest_range, agmax=magnitude_max)
    continuous = tp_desc.get_is_continuous()
    saturation_motor_min = params['quadcopter']['saturation_motor']
    logging.debug(f"[f_fwgym_get_env] Instantiating EnvID={env_id}, continuous={continuous}")
    if continuous:
        quadcopter=Quadcopter(T=tp_desc.qg_continuous.get_T_episode(), dt_commands=tp_desc.qg_continuous.get_dt_command(), dt=tp_desc.qg_continuous.get_dt(), saturation_motor_min=saturation_motor_min, aero=aero, windgust_generator=windgust_generator)
        #env.set_continuous(quadcopter=Quadcopter(T=tp_desc.qg_continuous.get_T_episode(), dt_commands=tp_desc.qg_continuous.get_dt_command(), dt=tp_desc.qg_continuous.get_dt()))
    else:
        quadcopter=Quadcopter(T=tp_desc.qg_episodic.get_T_episode(), dt_commands=tp_desc.qg_episodic.get_dt_command(), dt=tp_desc.qg_episodic.get_dt(), saturation_motor_min=saturation_motor_min, aero=aero, windgust_generator=windgust_generator)
        #env.set_episodic(quadcopter=Quadcopter(T=tp_desc.qg_episodic.get_T_episode(), dt_commands=tp_desc.qg_episodic.get_dt_command(), dt=tp_desc.qg_episodic.get_dt()))

    env = GymEnvBase.make(env_id=env_id, instance_index=instance_index, params=params, quadcopter=quadcopter, query_classes=query_classes, query_class=query_class, used_states=used_states)
    env.reset()
    return env 

def f_fwgym_get_action_noise(noise_dict, n_actions): 
    if noise_dict['name'] == 'OrnsteinUhlenbeck': 
        return OrnsteinUhlenbeckActionNoise(mean=float(noise_dict['mu'])*np.ones(n_actions), sigma=float(noise_dict['sigma']) * np.ones(n_actions))
    else: 
        raise RuntimeError(f"Unrecognized Noise Model {noise_dict['name']}")


def f_supports_n_envs(model_desc): 
    if model_desc.get_model_name() == "ppo": 
        return True 
    return False

