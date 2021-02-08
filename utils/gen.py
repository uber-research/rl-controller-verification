import matplotlib.pyplot as plt
import numpy as np
import logging

class TrainingParamsDict:
    def __init__(self, tp_dict):
        self.tp_dict = tp_dict


class SwitchParamsDict:
    def __init__(self, sp_dict):
        self.sp_dict = sp_dict

    def get_is_switch_active(self):
        return self.sp_dict['active'] == True

    def get_is_continuous(self):
        return self.sp_dict['target'] == "continuous"

    def get_time_perc(self):
        return float(self.sp_dict['time_perc'])


class QueryGenDict:
    def __init__(self, name, qg_dict):
        self.name = name
        self.qg_dict = qg_dict

    def get_T_episode(self):
        return self.qg_dict['T_episode']

    def get_dt(self):
        return self.qg_dict['dt']

    def get_dt_command(self):
        return self.qg_dict['dt_command']


class TrainingParamsDict:
    def __init__(self, tp_dict):
        self.tp_dict = tp_dict
        self.qg_continuous = QueryGenDict(
            name="continuous", qg_dict=self.tp_dict['query_generation']['continuous'])
        self.qg_episodic = QueryGenDict(
            name="episodic", qg_dict=self.tp_dict['query_generation']['episodic'])

    def get_is_continuous(self):
        return self.tp_dict['query_generation']['value'] == "continuous"

    def get_switch_params(self):
        return self.tp_dict['query_generation']['switch']


class EnvDict:
    def __init__(self, env_dict):
        self.env_dict = env_dict

    def get_env_id(self):
        return self.env_dict['value']

    def get_n_envs(self):
        return int(self.env_dict['params']['n_envs'])


class ModelDict:
    """ Abstraction Layer over the YAML training configuration.

    Structure can change over time but the information provided will stay the same.
    TODO: Implement Type Checking before returning value
    """

    def __init__(self, model_dict):
        self.model_dict = model_dict

    def get_model_name(self):
        return self.model_dict['name']

    def get_actor_feature_extractor_name(self):
        return self.model_dict['policy']['value']

    def get_actor_feature_extractor_type(self):
        return self.model_dict['policy']['type']['value']

    def get_actor_feature_extractor_architecture(self):
        return self.model_dict['policy']['type']['layers']

    def get_is_load(self):
        return self.model_dict['load']['value']

    def get_checkpoint_base_path(self):
        return self.model_dict['load']['checkpoint_base_path']

    def get_checkpoint_id(self):
        if not self.model_dict['load']['value']:
            return 0
        return self.model_dict['load']['checkpoint_id']

    def get_checkpoint_path(self):
        return f"{self.get_checkpoint_base_path()}/quadcopter-{self.get_checkpoint_id()}.pkl"

    def get_n_envs(self):
        return self.model_dict['params']['n_envs']


def f_iofsw_plot(x, y, x_ticks, y_ticks, title, label_x, label_y, filename):
    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    logging.info(f"Saving Figure to {filename}")
    plt.savefig(filename)


def sample(yaml_config):
    """ Function that implements a PDF according to the given YAML description and samples it
    """
    if yaml_config['pdf'] == 'none':
        return None
    elif yaml_config['pdf'] == 'uniform':
        f = np.random.uniform
        min_val = float(yaml_config['params'][0])
        max_val = float(yaml_config['params'][1])
        return f(min_val, max_val)
    else:
        raise RuntimeError(f"{yaml_config['pdf']} not supported")
