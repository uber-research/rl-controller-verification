import sys
import yaml
import logging
import tarfile

from pathlib import Path
from datetime import datetime as dt
from shutil import copyfile, rmtree
from types import SimpleNamespace

from training import eval_models
from infra.gcp.gcs import download_bucket
import doe


class TestingConfig:
    def __init__(self, config_path: str):
        self.now = dt.now()

        with open(config_path) as file_base_config:
            self.base = yaml.safe_load(file_base_config)

        self.filenames = self.base['filenames']
        self.results_base_dir = Path(self.filenames['results_base_dir'])
        with open(self.filenames['config_test']) as file_test_config:
            self.test = yaml.safe_load(file_test_config)['test']

        self.training_dir = Path(self.test['training_base_dir'])
        self.test_dir = self.experiment_path()
        self.plots_dir = self.test_dir / "figures"
        self.logs_dir = self.test_dir / "log"

    def load_training_config(self):
        """Find and load the training configuration

        The training directory might need to be downloaded from GCS first.
        """
        # Download training results from bucket if not available locally
        if not self.training_dir.exists() and "training_gcs_path" in self.test:
            folder = self.test["training_gcs_path"].replace('gs://atcp-data/', '').rstrip('/')
            logging.info(f'Download {folder} in {self.training_dir}')
            download_bucket(folder, destination=self.training_dir)

        # Find the training configuration file in the directory
        training_name = Path(self.filenames['config_training']).name
        for yaml_file in self.training_dir.glob('*.yaml'):
            if yaml_file.name == training_name:
                self.training_config_path = yaml_file
                break
        else:
            self.training_config_path = next(self.training_dir.glob('config*training_*.yaml'))

        # Load the yaml content
        logging.info(f'Found training config: {self.training_config_path}')
        with self.training_config_path.open() as file_training_config:
            self.training = yaml.safe_load(file_training_config)['training']

    def experiment_path(self) -> Path:
        if self.base['save']['with_datetime']:
            timestamp = self.now.strftime('%Y%m%d_%H%M%S')
            return self.training_dir / 'tests' / f"test_{timestamp}"
        else:
            return self.results_base_dir / 'testing' / self.filenames['output_relative_path']

    def create_output_dir_structure(self) -> None:
        for d in [self.test_dir, self.plots_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def copy_config(self) -> None:
        test_config = Path(self.filenames['config_test'])
        train_config = self.training_config_path
        copyfile(test_config, self.test_dir / test_config.name)
        copyfile(train_config, self.test_dir / train_config.name)

    def env_id(self):
        std = (self.test['env']['type'] == "standard")
        return self.training['env']['value'] if std else self.test['env']['value']

    def continuous(self) -> bool:
        if not self.test['mode']['type'] in ["old-continuous", "old-episodic"]:
            raise RuntimeError("Unsupported")
        return self.test['mode']['type'] == "old-continuous"

    def to_args(self):
        args = {
            'plots_dir'                 : str(self.plots_dir),
            'log_dir'                   : str(self.logs_dir),
            'training_base_path'        : str(self.training_dir),
            'env'                       : self.env_id(),
            'model'                     : self.training['model'],
            'num_iterations_checkpoint' : self.training['iterations_checkpoint'],
            'suffix'                    : self.test['suffix'],
            'n_episodes'                : self.test['n_episodes'],
            'continuous'                : self.continuous(),
            'save_plots'                : self.test['save_plots'],
            'start_index'               : self.test['checkpoints']['start_index'],
            'end_index'                 : self.test['checkpoints']['end_index'],
            'step'                      : self.test['checkpoints']['step'],
            # NOTE: This testing_params -> training_params is done on purpose as the environment performs its setup by using this field
            # NOTE: This also requires the following 2 structures must be always aligned: test.yaml.tpl testing_params and training.yaml.tpl training_params
            'testing_params'            : self.test['testing_params'],
            'used_states'               : self.training['used_states'],
            'query_classes'             : self.training['query_classes'],
            'query_class'               : self.test['testing_params']['query_class'],
            'eval_properties_observer'  : self.test['eval_properties_observer']
        }
        return SimpleNamespace(**args)

    def log_file(self) -> str:
        log_path = self.logs_dir / 'log.txt'
        return str(log_path)


if __name__ == "__main__":
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "run/config/default.yaml"
    config = TestingConfig(config_path)
    config.create_output_dir_structure()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config.log_file()),
            logging.StreamHandler()
        ]
    )

    # Prepare output directory
    config.load_training_config()
    config.copy_config()

    # Run testing
    test = eval_models.Test(config.to_args())
    test.run_test()

    # Compress results
    if config.base['save']['compress']:
        for ckpt in [d for d in config.logs_dir.iterdir() if d.is_dir()]:
            logging.info(f'Compress {ckpt}')
            with tarfile.open(f"{ckpt}.tar.gz", "w:gz") as tar:
                tar.add(ckpt, arcname=ckpt.name)
            rmtree(ckpt)

    # Done
    doe.IO.done(config.results_base_dir / 'testing')
