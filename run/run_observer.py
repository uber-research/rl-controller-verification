import hashlib
import logging
import sys
import yaml
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List

import infra.gcp.gcs as gcs
import doe
import properties_observer.perf_observer as perf_observer
import properties_observer.observe as observe


@dataclass
class EpisodesAggregator:
    """Episodes Aggregator

    Aggregate for N Episodes the 3xN traces (queries, commands, signals) in a
    checkpoint specific Dataframe and saves it in CSV format

    Attributes:
        dir_in: Directory where the various `episodes_{i}.{queries,commands,signals}.csv` can be found
        path_out: Fully qualified name for the CSV resulting from the aggregation
        nof_training_iterations: Number of training iterations, the checkpoint id is related
        prop_params: Dictionary of parameters for the Properties Observer
        load_signals: Parameters for loading the signales (t_is_hex, tlimit)
    """
    dir_in: Path
    path_out: Path
    nof_training_iterations: int
    prop_params: dict
    load_signals: dict

    def count_episodes(self) -> int:
        return len([e for e in Path(self.dir_in).iterdir() if e.is_file()]) // 3

    def aggregate(self, checkpoint_id: int) -> None:
        logging.info(f'Aggregate episodes from {self.dir_in} in {self.path_out}')

        def signal_dict(episode_id: int) -> dict:
            episode_basename = f"{self.dir_in}/episode_{episode_id}."
            logging.debug(f'Load signals from {episode_basename}')
            return perf_observer.load_signals(
                basename = episode_basename,
                t_is_hex = self.load_signals['t_is_hex'],
                tlimit = self.load_signals['tlimit']
            )

        observe.aggregate_episodes(
            checkpoint_params_dict = {
                'checkpoint_id': checkpoint_id,
                'nof_training_iterations': self.nof_training_iterations,
            },
            signals_dicts = [signal_dict(i) for i in range(self.count_episodes())],
            prop_params_dict = self.prop_params
        ).to_csv(self.path_out)


@dataclass
class CheckpointsAggregator:
    """Checkpoints Aggregator

    Aggregate N Checkpoints in an experiment specific Dataframe and saves it in CSV format

    Attributes:
        dir_in: Directory where the various `aggregated_cp.csv` can be found
        path_outdt: Fully qualified name for the CSV resulting from the aggregation
        experiment_params_dict: Experiment related info like the ID and similar
    """
    dir_in: Path
    path_out: Path
    experiment_params_dict: dict

    def aggregate(self) -> None:
        def to_pandas_format(d: dict) -> dict:
            return {k: [d[k]] for k in d}

        checkpoints_csv_list = sorted([csv for csv in Path(self.dir_in).iterdir() if csv.is_file()])
        logging.debug(checkpoints_csv_list)
        if not checkpoints_csv_list:
            logging.warning("no checkpoints to aggregate")
            return

        logging.info(f'Aggregate checkpoints from {self.dir_in} in {self.path_out}')
        observe.aggregate_checkpoints(
            experiment_params_dict = to_pandas_format(self.experiment_params_dict),
            checkpoint_dataframes = [pd.read_csv(csv) for csv in checkpoints_csv_list]
        ).to_csv(self.path_out)


@dataclass
class ExperimentsAggregator:
    """Experiments Aggregator

    Aggregate N Experiments in a Round specific Datafrane in CSV format

    Attributes:
        dir_in: Directory where the various `aggregated_cp.csv` can be found
        path_out: Fully qualified name for the CSV resulting from the aggregation
    """
    dir_in: Path
    path_out: Path

    def aggregate(self) -> None:
        experiments_csv_list = sorted([csv for csv in Path(self.dir_in).iterdir() if csv.is_file()])
        logging.debug(experiments_csv_list)
        if not experiments_csv_list:
            logging.warning("no experiments to aggregate")
            return

        logging.info(f'Aggregate experiments from {self.dir_in} in {self.path_out}')
        observe.aggregate_experiments(
            experiment_dataframes=[pd.read_csv(csv) for csv in experiments_csv_list]
        ).to_csv(self.path_out)


class ObserverConfig:
    def __init__(self, config_path):
        with open(config_path) as file_base_config:
            self.base = yaml.safe_load(file_base_config)

        self.filenames = self.base['filenames']
        self.results_base_dir = Path(self.filenames['results_base_dir'])
        with open(self.filenames['config_observer']) as file_observer_config:
            self.observer = yaml.safe_load(file_observer_config)['prop_obs']

    def load_testing_config(self):
        """Find and load the testing configuration

        The testing directory might need to be downloaded from GCS first.
        """
        testing_dir = self.testing_dir_path()
        logging.info(f'Look for configurations in {testing_dir}')

        # Download testing results from bucket if not available locally
        if not testing_dir.exists() and "testing_gcs_path" in self.observer:
            folder = self.observer["testing_gcs_path"].replace('gs://atcp-data/', '').rstrip('/')
            logging.info(f'Download {folder} in {testing_dir}')
            gcs.download_bucket(folder, destination=testing_dir)
            gcs.extract_all_tar_gz(testing_dir/'log')

        if not testing_dir.exists():
            logging.critical(f'Cannot load testing results.')
            logging.critical(f'Check that "{testing_dir}" exist or set "testing_gcs_path" in observer config to download it from GCS.')
            raise RuntimeError('cannot load testing results')

        # Find the training and testing configuration files in the directory
        training_name = Path(self.filenames['config_training']).name
        testing_name = Path(self.filenames['config_test']).name
        found_training, found_testing = False, False
        for yaml_file in testing_dir.glob('*.yaml'):
            if yaml_file.name == training_name:
                self.training_config_path = yaml_file
                found_training = True
            if yaml_file.name == testing_name:
                self.testing_config_path = yaml_file
                found_testing = True
        if not found_training:
            self.training_config_path = next(testing_dir.glob('*training_*.yaml'))
        if not found_testing:
            self.testing_config_path = next(testing_dir.glob('*test_*.yaml'))

        # Load the yaml content
        logging.info(f'Found training config: {self.training_config_path}')
        logging.info(f'Found testing config: {self.testing_config_path}')
        with self.training_config_path.open() as file_training_config:
            self.training = yaml.safe_load(file_training_config)['training']
        with self.testing_config_path.open() as file_testing_config:
            self.testing = yaml.safe_load(file_testing_config)['test']

    def exp_params(self) -> dict:
        layers = self.training["model"]["policy"]["type"]["layers"]
        params = {
            "algo"                              : self.training["model"]["name"],
            "query_class"                       : self.testing["testing_params"]["query_class"],
            "nb_layers"                         : len(layers),
            "nb_neurons"                        : layers[0],
            "test_aero_enabled"                 : self.testing["testing_params"]["aero"]["enabled"],
            "test_windgust_magnitude_max"       : self.testing["testing_params"]["aero"]["windgust"]["magnitude_max"],
            "test_saturation_motor"             : self.testing["testing_params"]["quadcopter"]["saturation_motor"],
            "training_saturation_motor"         : self.training["training_params"]["quadcopter"]["saturation_motor"],
            "pid_rates"                         : self.testing["testing_params"]["pid_rates"],
            "test_pid_thrust"                   : self.testing["testing_params"]["pid_thrust"],
            "training_pid_thrust"               : self.testing["testing_params"]["pid_thrust"],
            "training_aero_enabled"             : self.training["training_params"]["aero"]["enabled"],
            "training_windgust_magnitude_max"   : self.training["training_params"]["aero"]["windgust"]["magnitude_max"],
        }
        # add the used states booleans in the dict
        all_states = "thrust p q r e_p e_q e_r".split(' ')
        for state in all_states:
            params[state] = state in self.training["used_states"]
        return params

    def aggregate(self, key: str) -> Any:
        return self.observer['aggregate'].get(key)

    def testing_dir_path(self) -> Path:
        return Path(self.base['filenames']['results_base_dir'], self.aggregate('testing_base_dir'))

    def checkpoint(self, id: int) -> str:
        return f"checkpoint_{id}"

    def output_dir_path(self) -> Path:
        base_dir = Path(self.base['filenames']['results_base_dir'])
        return base_dir / 'observer' / self.base['filenames']['output_relative_path']

    def aggregated_checkpoints_dir_path(self) -> Path:
        return self.output_dir_path() / 'aggregated_cps'

    def aggregated_experiments_dir_path(self) -> Path:
        return self.output_dir_path() / 'aggregated_exp'

    def input_checkpoint_path(self, id) -> Path:
        return self.testing_dir_path() / 'log' / self.checkpoint(id)

    def output_checkpoint_csv(self, id) -> Path:
        return self.aggregated_checkpoints_dir_path() / f'{self.checkpoint(id)}.csv'

    def output_experiment_csv(self) -> Path:
        # hash training and testing config to name experiment file
        m = hashlib.sha256()
        m.update(yaml.dump(self.training, encoding='utf-8'))
        m.update(yaml.dump(self.testing, encoding='utf-8'))
        return self.aggregated_experiments_dir_path() / f'experiment_{m.hexdigest()[:7]}.csv'

    def output_all_csv(self) -> Path:
        return self.output_dir_path() / 'aggregated_all.csv'

    def list_checkpoint_dirs(self) -> List[Path]:
        logdir = self.testing_dir_path() / 'log'
        return [d for d in logdir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]

    def list_checkpoint_ids(self) -> List[int]:
        checkpoint_id = self.aggregate('checkpoint_id')
        all_ids = [int(d.name.replace('checkpoint_', '')) for d in self.list_checkpoint_dirs()]
        if checkpoint_id is not None and int(checkpoint_id) >= 0:
            if checkpoint_id not in all_ids:
                raise RuntimeError(f'checkpoint id {checkpoint_id} is not available.')
            return [checkpoint_id]
        return sorted(all_ids)

    def create_output_dir_structure(self) -> None:
        dirs = [self.output_dir_path()]
        if self.aggregate('episodes'):
            dirs.append(self.aggregated_checkpoints_dir_path())
        if self.aggregate('checkpoints'):
            dirs.append(self.aggregated_experiments_dir_path())
        [d.mkdir(parents=True, exist_ok=True) for d in dirs]


if __name__ == "__main__":
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "run/config/default.yaml"
    config = ObserverConfig(config_path)
    config.create_output_dir_structure()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Run aggregations
    if config.aggregate('episodes') or config.aggregate('checkpoints'):
        config.load_testing_config()

    if config.aggregate('episodes'):
        [EpisodesAggregator(
            config.input_checkpoint_path(checkpoint_id),
            config.output_checkpoint_csv(checkpoint_id),
            config.training['iterations_checkpoint'] * checkpoint_id,
            config.observer['po_params'],
            config.observer['load_signals']
        ).aggregate(checkpoint_id)
        for checkpoint_id in config.list_checkpoint_ids()]

    if config.aggregate('checkpoints'):
        CheckpointsAggregator(
            config.aggregated_checkpoints_dir_path(),
            config.output_experiment_csv(),
            config.exp_params()
        ).aggregate()

    if config.aggregate('experiments'):
        ExperimentsAggregator(
            config.aggregated_experiments_dir_path(),
            config.output_all_csv(),
        ).aggregate()

    # Done
    results_base_dir = Path(config.base['filenames']['results_base_dir'])
    doe.IO.done(results_base_dir / 'observer')
