import sys
import yaml
import logging
from pathlib import Path
from training import train, parsers
import doe

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "run/config/default.yaml"

    with open(config_path) as file_base_config:
        base_config = yaml.safe_load(file_base_config)

    with open(base_config['filenames']['config_training']) as config_file:
        config = yaml.safe_load(config_file)['training']

    args = parsers.training_args_from_configs(base_config, config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{args.log_dir}/log.txt"),
            logging.StreamHandler()
        ]
    )
    training = train.Training()
    training.run_training(args)
    doe.IO.done(base_config['filenames']['results_base_dir'])
