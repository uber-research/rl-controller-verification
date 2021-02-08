#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numba as nb
from numpy import inf
from numba import float64, int64, types
from numba.experimental import jitclass
import itertools
from pwcsignalnumba import PwcSignal
from props import *


def load_signals(basename, t_is_hex=False, tlimit=None):
    """
    Loads quadcopter signals from Dataframes in CSV format.
    All signals are assumed to be real valued signals in hex format.
    
    :param basename:
        File prefix. It is extended with suffixes 'signals.csv', 'commands.csv', queries.csv'
    
    :param t_is_hex: 
        Set to True if the time steps column 't' is hex format, False if in decimal format.
    
    :param tlimit:
        if None all samples are loaded
        if not None sample values after this time limit get discarded.
        
    :returns: 
        A dict containing raw signals as np.arrays under keys
        p, q, r, query_p, query_q, query_r, cmd_phi, cmd_theta, cmd_psi
    """
    def fromhex(x): return float.fromhex(x)

    # result dictionary
    res = {}

    # load signals
    signals = pd.read_csv(f'{basename}signals.csv')
    t = signals['t'].apply(fromhex).values if t_is_hex else signals['t'].values
    p = signals['p'].apply(fromhex).values
    q = signals['q'].apply(fromhex).values
    r = signals['r'].apply(fromhex).values
    res['t'] = t
    res['p'] = p
    res['q'] = q
    res['r'] = r

    # load queries
    queries = pd.read_csv(f'{basename}queries.csv')
    query_t = queries['t'].apply(fromhex).values if t_is_hex else queries['t'].values
    query_p = queries['query_p'].apply(fromhex).values
    query_q = queries['query_q'].apply(fromhex).values
    query_r = queries['query_r'].apply(fromhex).values
    # resample on the signals timebase
    sig_query_p = PwcSignal(query_t, query_p, -inf)
    sig_query_q = PwcSignal(query_t, query_q, -inf)
    sig_query_r = PwcSignal(query_t, query_r, -inf)
    query_p = np.array([sig_query_p.apply(each) for each in t])
    query_q = np.array([sig_query_q.apply(each) for each in t])
    query_r = np.array([sig_query_r.apply(each) for each in t])
    res['query_p'] = query_p
    res['query_q'] = query_q
    res['query_r'] = query_r

    # enfore tlimit if defined
    if tlimit is not None:
        mask = np.array([True for each in t]) if tlimit is None else t < tlimit
        res = {k: res[k][mask] for k in res}
    
    return res


class MinMaxAvg:
    """
    Tracks min, max, average on a stream of values.
    """
    def __init__(self, name='name not set'):
        self.name = name
        self.count = 0
        self.min = inf
        self.max = -inf
        self.avg = 0.0
        self.cols = [f'{self.name} count', f'{self.name} min', f'{self.name} max', f'{self.name} avg']

    def update(self, x):
        """
        Adds a new sample to the stats.
        """
        prev_count = self.count
        self.count += 1
        assert prev_count < self.count, f"integer overflow in MinMaxAvg({self.name})"
        self.min = min(x, self.min)
        self.avg += (x - self.avg) / (self.count) # yields x on first update
        self.max = max(x, self.max)
    
    def reset(self):
        """
        Resets internal counters to their initial state.
        """
        self.count = 0
        self.min = inf
        self.max = -inf
        self.avg = 0.0
    
    def get(self):
        """
        Returns a snapshot of current stats.
        """
        return [self.count, self.min, self.max, self.avg]
    
    def get_cols(self):
        """
        Returns columns names for stats.
        """
        return self.cols.copy()

    def __str__(self):
        return f'{self.name}: #{self.count} [{self.min:.5}, {self.max:.5}] avg {self.avg:.5}'


class PerfObserver(object):
    """
    Observer of STL properties for angular query tracking on a single axis.
    """

    @staticmethod
    def default_prop_params():
        """
        Returns a default parameters dictionary.
        """
        res = {
            # lookahead for query stability in secs.
            'query_lh': 0.01,
            # settling time in secs.
            'tset': 0.25,
            # stability time in secs.
            'tstab': 0.5,
            # stability margin for the query (absolute).
            'query_diam': 0.05,
            # 10% offset error percentage relative to stepsize.
            'offset_pct': 0.05,
            # 10% overshoot error percentage relative to stepsize.
            'overshoot_pct': 0.1,
            # perctentage for the time to reach prop (absolute).
            'rising_time_pct': 0.05,
        }
        return res


    def __init__(self, max_sig_len, prop_params):
        """
        :param max_sig_len: 
            size of the traces for which this observer is configured in number of samples.
        
        :param prop_params: 
            static property parameters.
        """

        # this observer is
        self.max_sig_len = max_sig_len

        # unpack params
        self.prop_params     = prop_params.copy()
        self.query_lh        = self.prop_params['query_lh']
        self.tset            = self.prop_params['tset']
        self.tstab           = self.prop_params['tstab']
        self.query_diam      = self.prop_params['query_diam']
        self.offset_pct      = self.prop_params['offset_pct']
        self.overshoot_pct   = self.prop_params['overshoot_pct']
        self.rising_time_pct = self.prop_params['rising_time_pct']

        # Property stats counters
        self.nof_episode       = 0
        self.nof_stable        = 0
        self.episode_len       = MinMaxAvg('episode_len')
        self.offset_all        = MinMaxAvg('offset_all')
        self.offset_all_pct    = MinMaxAvg('offset_all_rel')
        self.offset_bad        = MinMaxAvg('offset_bad')
        self.offset_bad_pct    = MinMaxAvg('offset_bad_rel')
        self.overshoot_all     = MinMaxAvg('overshoot_all')
        self.overshoot_all_pct = MinMaxAvg('overshoot_all_rel')
        self.overshoot_bad     = MinMaxAvg('overshoot_bad')
        self.overshoot_bad_pct = MinMaxAvg('overshoot_bad_rel')
        self.rising_time_all   = MinMaxAvg('rising_time_all')
        self.rising_time_bad   = MinMaxAvg('rising_time_bad')
        
        self.all_stats = [
            self.episode_len,
            self.offset_all,
            self.offset_all_pct,
            self.offset_bad,
            self.offset_bad_pct,
            self.overshoot_all,
            self.overshoot_all_pct,
            self.overshoot_bad,
            self.overshoot_bad_pct,
            self.rising_time_all,
            self.rising_time_bad,           
        ]

        self.cols = \
            [*self.prop_params] + \
            ['nof_episode', 'nof_stable'] + \
            [col for stat in self.all_stats for col in stat.get_cols()]

        # arrays for storing property values for plots
        self.t = np.zeros(self.max_sig_len)
        self.x = np.zeros(self.max_sig_len)
        self.q = np.zeros(self.max_sig_len)
        self.stable = np.zeros(self.max_sig_len)
        self.offset = np.zeros(self.max_sig_len)
        self.offsetval = np.zeros(self.max_sig_len)
        self.overshoot = np.zeros(self.max_sig_len)
        self.overshootval = np.zeros(self.max_sig_len)
        self.stepsize = np.zeros(self.max_sig_len)
        self.rising_time = np.zeros(self.max_sig_len)

        # index trackers for subrange plotting
        self.min_idx = self.max_sig_len + 1
        self.max_idx = -1

    def reset_idx_range(self):
        """
        Resets subrange trackers (for plots).
        """
        self.min_idx = self.max_sig_len + 1
        self.max_idx = -1

    def reset_stats(self):
        """
        Resets all stat counters.
        """
        self.nof_episode       = 0
        self.nof_stable        = 0

        for each in self.all_stats:
            each.reset()

    def get_stats_col_names(self):
        """
        Returns results dataframe column names.
        """
        return self.cols.copy()

    def get_stats_row(self):
        """
        Returns results datafram row.
        """
        res = [*self.prop_params.values()] +\
            [self.nof_episode, self.nof_stable] +\
            [val for stat in self.all_stats for val in stat.get()]
        return res

    def get_stats_dict(self):
        """
        Returns the dictionary of current params/stats.
        """
        return { k: v for k, v in zip(self.cols, self.get_stats_row()) }

    def __str__(self):
        """
        Prints stats dictionary.
        """
        tmp = self.get_stats_dict()
        return "\n".join([f'{k}: {tmp[k]}' for k in tmp])

    def sig_plot(self):
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.q[self.min_idx:self.max_idx],
            'g',
            label='query')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.x[self.min_idx:self.max_idx],
            'b',
            label='signal')
        plt.xlabel('time (s)')
        plt.ylabel('angular rate (rad/s)')
        plt.legend()
        plt.title(f'query and signal ')
        plt.show()

    def stable_plot(self):
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.q[self.min_idx:self.max_idx],
            "g",
            label='query')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.stable[self.min_idx:self.max_idx],
            'm--',
            label='stable')
        plt.xlabel('time (s)')
        plt.ylabel('angular rate (rad/s)')
        plt.legend()
        plt.title(f'query is stable on [0.0s, {self.tstab}s]')
        plt.show()

    def stepsize_plot(self):
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.q[self.min_idx:self.max_idx],
            "g",
            label='query')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.stepsize[self.min_idx:self.max_idx],
            'm--',
            label='stepsize')
        plt.xlabel('time (s)')
        plt.ylabel('angular rate (rad/s)')
        plt.legend()
        plt.title(f'query stepsize when stable on [0.0s, {self.tstab}s]')
        plt.show()

    def offset_plot(self):
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.q[self.min_idx:self.max_idx],
            'g',
            label='query')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.x[self.min_idx:self.max_idx],
            'b',
            label='signal')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.offsetval[self.min_idx:self.max_idx],
            'm',
            label='offset')
        plt.xlabel('time (s)')
        plt.ylabel('angular rate (rad/s)')
        plt.legend()
        plt.title(f'max offset on [{self.tset}s, {self.tstab}s] w/ stable query on [0.0s, {self.tstab}s]')
        plt.show()

    def overshoot_plot(self):
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.q[self.min_idx:self.max_idx],
            'g', 
            label='query')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.x[self.min_idx:self.max_idx],
            'b',
            label='signal')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.overshootval[self.min_idx:self.max_idx],
            'm',
            label='overshoot')
        plt.xlabel('time (s)')
        plt.ylabel('angular rate (rad/s)')
        plt.legend()
        plt.title(f'max overshoot on [0.0s, {self.tset}s] w/ stable query on [0.0s, {self.tstab}s]')
        plt.show()

    def rising_time_plot(self):
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.q[self.min_idx:self.max_idx],
            'g',
            label='query')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.x[self.min_idx:self.max_idx],
            'b',
            label='signal')
        plt.plot(
            self.t[self.min_idx:self.max_idx],
            self.rising_time[self.min_idx:self.max_idx],
            'm',
            label='t. to reach q')
        plt.xlabel('time (s)')
        plt.ylabel('angular rate (rad/s)')
        plt.legend()
        plt.title(f'time to reach a stable query within {self.rising_time_pct}% on [0.0s, {self.tstab}s]')
        plt.show()

    def all_plots(self):
        self.sig_plot()
        self.stable_plot()
        self.stepsize_plot()
        self.offset_plot()
        self.overshoot_plot()
        self.rising_time_plot()

    def observe(self, t, x, q, save_eval=False):
        """
        :param t: time breakpoints
        :param x: angular velocity signal
        :param q: angular velocity query
        :param save_eval: set to true to save evaluation data for plotting methods
        """
        # allow signals shorter than max_sig_len.
        episode_len = len(t)
        assert episode_len <= self.max_sig_len
        assert episode_len == len(x)
        assert episode_len == len(q)

        self.nof_episode += 1
        self.episode_len.update(episode_len)

        # instanciate the observer
        obs = AllProps(
            t, t, x, q, 
            self.tset, self.tstab, self.query_lh, self.query_diam, 
            self.offset_pct, self.overshoot_pct, self.rising_time_pct
        )

        # compute props/stats, only for samples that have enough
        # negative and positive lookahead in the trace, ie such that:
        # t[i] - t[0] > query_lh
        # t[-1] - t[i] > tstab

        # reset indices
        self.reset_idx_range()

        for i, each in enumerate(t):

            if  each - t[0] > self.query_lh and\
               t[-1] - each > self.tstab:

                # update indices
                self.min_idx = min(self.min_idx, i)
                self.max_idx = max(self.max_idx, i)

                # observe
                stable, stepsize, offset, offsetval, overshoot, overshootval, rising_time =\
                    obs.apply(each)

                # save for plots
                if save_eval:
                    self.t[i] = each
                    self.q[i] = q[i]
                    self.x[i] = x[i]
                    self.stable[i] = stable
                    self.offset[i] = offset
                    self.offsetval[i] = offsetval
                    self.overshoot[i] = overshoot
                    self.overshootval[i] = overshootval
                    self.stepsize[i] = stepsize
                    self.rising_time[i] = rising_time

                # update stats
                if stable:
                    self.nof_stable += 1
                    abs_stepsize = abs(stepsize)
                    self.overshoot_all.update(overshootval)
                    self.offset_all.update(offsetval)
                    if abs_stepsize != 0.0:
                        self.overshoot_all_pct.update(100.0 * overshootval / abs_stepsize)
                        self.offset_all_pct.update(100.0 * offsetval / abs_stepsize)

                    if overshoot:
                        self.overshoot_bad.update(overshootval)
                        if abs_stepsize != 0.0:
                            self.overshoot_bad_pct.update(100.0 * overshootval / abs_stepsize)

                    if offset:
                        self.offset_bad.update(offsetval)
                        if abs_stepsize != 0.0:
                            self.offset_bad_pct.update(100.0 * offsetval / abs_stepsize)

                    if rising_time >= 0.0:
                        self.rising_time_all.update(rising_time)
                    else:
                        self.rising_time_bad.update(-np.inf)
