import os
import yaml
import numpy as np

from pathlib import Path

import architectures.export
from utils.params_nn import get_stable_baseline_file_params


def get_configs():
    with open("config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    with open(base_config['filenames']['config_export'], 'r') as f:
        config_export = yaml.safe_load(f)

    exp_dir = Path(config_export['input']['experiment_dir'])
    training_conf_path = exp_dir / base_config['filenames']['config_training']
    with open(training_conf_path) as f:
        config_training = yaml.safe_load(f)

    return base_config, config_export, config_training


def get_params(config_export):
    exp_dir = Path(config_export['input']['experiment_dir'])
    subdir = exp_dir / config_export['input']['stable_baseline_checkpoint_subdir']
    p = subdir / config_export['input']['stable_baseline_checkpoint_filename']
    return get_stable_baseline_file_params(p)


def export_sherlock(config_export, config_training):
    exp_dir = Path(config_export['input']['experiment_dir'])
    subdir = exp_dir / config_export['output']['subdir']
    subdir.mkdir(exists_ok=True)

    basename = config_export['input']['stable_baseline_checkpoint_filename']
    output_path = subdir / f"{basename}.sherlock"
    print(f"Saving Sherlock Export in {output_path}")
    with open(output_path, 'w') as f:
        f.write(architectures.export.get_sherlock_format(
            config_training['training']['model'], params))


def debug_print_tensors(config_export, params):
    np.set_printoptions(precision=3)
    for tensor_name in config_export['debug']['show_tensors']:
        tensor_vals = np.array(params[tensor_name])
        print(f"-----------")
        print(f"Tensor {tensor_name} dim is {tensor_vals.shape}")
        print(f"{tensor_vals}")


if __name__ == "__main__":
    base_config, config_export, config_training = get_configs()
    params = get_params(config_export)

    if config_export['output']['console']:
        print(params)

    if config_export['output']['file']:
        export_sherlock(config_export, config_training)

    if config_export['debug']['is_active']:
        debug_print_tensors(config_export, params)
