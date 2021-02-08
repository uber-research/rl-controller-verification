#!/usr/bin/env python
# coding: utf-8

from properties_observer.perf_observer import load_signals, PerfObserver
import pandas as pd


def aggregate_experiments(experiment_dataframes):
    """
    :param checkpoint_dataframes:
        Dataframes of all experiments of the round.
    """
    # aggregate all experiments data add columns corresponding to dictionary keys
    res = None
    for df in experiment_dataframes:
        if res is None:
            res = df
        else:
           res = res.append(df, ignore_index=True)
    return res

def aggregate_checkpoints(experiment_params_dict, checkpoint_dataframes):
    """
    :param experiment_params_dict:

        Parameters of the experiment stored in a dictionary as follows:
        {
            'param1' : [value], # value needs to be wrapped in a list
            'param2' : [value],
            'param3' : [value],
            'param4' : [value],
        }

        A flat and tidy dictionary, with scalar keys wrapped in lists.

    :param checkpoint_dataframes:
        Dataframes of all checkpoints of the experiment.

    :returns:
        A dataframe resulting from appending all checkpoint dataframes into a single dataframe, 
        augmented with leading colums describing experiment parameters.
    """
    # aggregate episode_dataframes add columns corresponding to dictionary keys
    res = None
    for df in checkpoint_dataframes:
        if res is None:
            res = df
        else:
           res = res.append(df, ignore_index=True)
    extra_cols = pd.DataFrame(experiment_params_dict, index = res.index)
    res = pd.concat([extra_cols, res], axis = 1)
    return res

def aggregate_episodes(checkpoint_params_dict, signals_dicts, prop_params_dict):
    """
    Aggregates episodes of a given checkpoint in a Dataframe

    Args:    
        checkpoint_params_dict      (dict)          :       dictionary containing keys
                                                            - 'checkpoint_id': [int or string] 
                                                            - 'nof_training_iterations': [int]
        
        signals_dicts               (list[dict])    :       list of signals dictionaries

        prop_params_dict            (dict)          :       Static property parameter values for the observer.

    Returns: 
        Returns a DataFrame containg observations metrics for axes p, q, r of the controller.
        Columns of the DataFrame are as follows: [checkpoint_id, nof_training_iterations, episode_id, axis, *prop_params_dict, *observer_stats]
    """
    res = None
    # Iterates over the list of results of `load_signals()` API
    for i, signals_dict in enumerate(signals_dicts):
        tmp = observe_episode(i, signals_dict, prop_params_dict)
        if res is None:
            res = tmp
        else:
           res = res.append(tmp, ignore_index=True)
    extra_cols = pd.DataFrame(checkpoint_params_dict, index = res.index)
    res = pd.concat([extra_cols, res], axis = 1)
    return res

def observe_episode(episode_id, signals_dict, prop_params_dict):
    """
    Runs the observer on the given signals with given parameters and 
    returns a dataframe containing observations for all axes.

    Args:
        episode_id          (int)       :           Episode ID
        signals_dict        (dict)      :           Dictionary containing signals as numpy float64[:] under keys  't', 'p', 'q', 'r', 'query_p', 'query_q', 'query_r'
                                                    It should be the result of `load_signals()` API
        prop_params_dict    (dict)      :           Static property parameter values for the observer

    Returns: 
        Returns a DataFrame containg observations metrics for axes p, q, r of the controller.
        Columns of the DataFrame are as follows: [episode_id, axis, *prop_params, *observer_stats]
    """
    t = signals_dict['t']
    sig_len = len(t)
    axes = ['p','q','r']
    cols = None
    rows = []
    for axis in axes:
        x = signals_dict[axis]
        q = signals_dict['query_' + axis]
        obs = PerfObserver(sig_len, prop_params_dict)
        obs.observe(t, x, q)
        if cols is None:
            cols = obs.get_stats_col_names()
            cols.insert(0, 'axis')
            cols.insert(0, 'episode_id')
        row = obs.get_stats_row()
        row.insert(0, axis)
        row.insert(0, episode_id)
        rows.append(row)

    df = pd.DataFrame(rows, columns = cols)
    return df
