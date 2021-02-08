import argparse

from types import SimpleNamespace

from utils.in_notebook import is_in_notebook
from training import plots

def default_plots_args():
    args = {
        'env'           : 0,
        'n_episodes'    : 100,
        'plots_dir'     : 'results'
    }
    return SimpleNamespace(**args)

def cli_plots_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--env', dest='env', default=0, type=int)
    parser.add_argument('--n_episodes', dest='n_episodes', default=100, type=int)
    parser.add_argument('--plots_dir', dest='plots_dir', default="results")
    return parser.parse_args()

if __name__ == "__main__":
    args = cli_plots_args() if is_in_notebook() else default_plots_args()
    p = plots.Plots()
    p.run_plots(args)
