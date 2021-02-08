import os

from pathlib import Path
from shutil import copyfile, rmtree
from datetime import datetime as dt
from types import SimpleNamespace

import buildinfo

def create_experiment_dir_structure(base_dir, dir_names):
    p = Path(base_dir)
    p.mkdir(exist_ok=True, parents=True)
    [d.mkdir(exist_ok=True) for d in [Path(f"{base_dir}/{d}") for d in dir_names]]

def get_merged(x, y):
    """ Merges to 2 dicts
    """
    res = x.copy()
    for k in res.keys():
        if k in y.keys():
            if isinstance(res[k], dict): res[k] = get_merged(x=res[k], y=y[k])
            else: res[k] = y[k]
    return res


def training_args_from_configs(base_config, config, debug_info=None):
    """ Function transforming the YAML config into params for the training 

    Args 
    base_config         :       Loaded YAML of the base config 
    config              :       Loaded YAML of the training config 
    skip_fs_ops         :       Avoids File System Operations. Mainly used during Unit Testing  
    """

    def experiment_path():
        filenames = base_config['filenames']
        if config['save']['with_datetime']:
            now = dt.now()
            return f"{filenames['results_base_dir']}/experiment_{now.strftime('%Y%m%d_%H%M')}"
        else:
            return f"{filenames['results_base_dir']}/{filenames['output_relative_path']}"

    def copy_config_file(experiment_dir):
        src = base_config['filenames']['config_training']
        dest = f"{experiment_dir}/{os.path.basename(src)}"
        copyfile(src, dest)

    def create_meta_info_file(experiment_dir):
        meta_info_filename = f"{experiment_dir}/{base_config['filenames']['meta_info_filename']}"
        with open(meta_info_filename, 'w') as f:
            f.write(f"{buildinfo.buildinfo}\n")
            f.close()

    def evaluation_env(config):
        temp = config['evaluation']['env']['type']
        if temp == "standard":
            return config['env']['value']
        elif temp == "custom":
            return config['evaluation']['env']['value']
        else:
            raise RuntimeError("Unsupported")

    def evaluation_mode(config):
        temp = config['evaluation']['mode']['type']
        if temp == "old-continuous":
            return True
        elif temp == "old-episodic":
            return False
        else:
            raise RuntimeError("Unsupported")

    base_dir = experiment_path()
    create_experiment_dir_structure(base_dir, ["figures", "models", "exports", "log", "runs"])

    if debug_info is None: 
        copy_config_file(base_dir)
        create_meta_info_file(base_dir)


    args = {
        'save_with_datetime'            : config['save']['with_datetime'],
        'base_dir'                      : base_dir,
        'plots_dir'                     : f"{base_dir}/figures",
        'models_dir'                    : f"{base_dir}/models",
        'exports_dir'                   : f"{base_dir}/exports",
        'log_dir'                       : f"{base_dir}/log",
        'log_dir_tensorboard'           : f"{base_dir}/runs",
        'save_tf_checkpoint'            : config['save']['tf_checkpoint'],
        'save_as_tf'                    : config['save']['as_tf'],
        'step'                          : config['step'],
        'training_params'               : config['training_params'],
        'testing_params'                : get_merged(x=config['training_params'], y=config['testing_params']),
        'env'                           : config['env'],
        'logging'                       : config['logging'],
        'verbose'                       : config['logging']['stable_baselines']['verbosity'],
        'save_plots'                    : config['save_plots'],
        'suffix'                        : config['suffix'],
        'activation'                    : config['activation'],
        'activation_end_of_actor'       : config['activation_end_of_actor'],
        'action_noise'                  : config['action_noise'],
        'model'                         : config['model'],
        'n_steps'                       : config['n_steps'],
        'iterations_checkpoint'         : int(config['iterations_checkpoint']),
        'debug_is_active'               : config['debug']['active'],
        'debug_run_training_loop'       : config['debug']['run_training_loop'],
        'debug_model_describe'          : config['debug']['describe'],
        'debug_try_save_all_vars'       : config['debug']['try_save']['all_variables'],
        'debug_try_save_trainable_vars' : config['debug']['try_save']['trainable_variables'],
        'debug_try_save_graph'          : config['debug']['try_save']['graph'],
        'debug_try_save_weights'        : config['debug']['try_save']['weights'],
        'debug_show_tensors_active'     : config['debug']['show_tensors']['active'],
        'debug_show_tensors_list'       : config['debug']['show_tensors']['list'],
        'query_classes'                 : config['query_classes'],
        'query_class'                   : config['query_class'],
        'debug_info'                    : debug_info,
        'used_states'                   : config['used_states']
    }
    if args['model']['name'] == "ppo":
        args['iterations_checkpoint'] /= 100
    elif args['model']['name'] == "trpo":
        args['iterations_checkpoint'] /= 1000
    return SimpleNamespace(**args)
