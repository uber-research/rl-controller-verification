#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 23:10:39 2020

@author: arthur.gold
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def trim(df, t):
    return df.loc[df["t"]<10]

path_input = Path("network_selections")
path_output = Path("generated_figures")

def create_plot(path_in, path_out):
    fig, axes = plt.subplots(1, 1, sharey=True, dpi=150)
    params = ["p", "q", "r"]
    names = ["roll rate", "pitch rate", "yaw rate"]
    for i in range(1):
        query = pd.read_csv(path_in/"episode_0.queries.csv")
        signals = pd.read_csv(path_in/"episode_0.signals.csv")
        query = query.applymap(lambda x: float.fromhex(x))
        signals = signals.applymap(lambda x: float.fromhex(x))
        query = trim(query, 10)
        signals = trim(signals, 10)
        axes.plot(signals["t"],signals[params[i]], label=names[i])
        axes.plot(query["t"],query[f"query_{params[i]}"], label=f"query_{params[i]}")
        axes.set_ylabel("amgular rate (rad/s)")
        axes.set_xlabel("time (s)")
        axes.legend(loc="lower right")
    plt.savefig(path_out)

def figure3():
    p = Path("data/PID/better,main/log")
    create_plot(p, path_output / "figure3.png")
    
def figure4():
    p = Path("data/sac_16_16_cp_30" )
    create_plot(p, path_output / "figure4.png")

def figure5():
    p = Path(path_input / "sac_32_32.csv")
    data = pd.read_csv(p)
    data["nof_training_iterations"] /= 1000000
    performances = ["OK rising t.", "OK off.", "OK overshoot"]
    for perf in performances:
        data[perf] *= 100
    data = data.set_index("nof_training_iterations")   
    over = data["OK overshoot"]
    offset = data["OK off."]
    rising_time = data["OK rising t."]
    fig, ax1 = plt.subplots(dpi=150)
    color="tab:green"
    ax1.set_xlabel('number of training iterations (in millions)')
    ax1.set_ylabel('% of success')
    ax1.plot(over, color="orange", label="OK overshoot")
    ax1.plot(offset, color="blue", label="OK offset")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.plot(np.nan, color=color, label = "OK rising time")
    ax1.grid()
    ax1.legend(loc=0)
    ax1.plot(rising_time, color=color, label="OK rising time")
    ax1.tick_params(axis='y')
    fig.tight_layout()
    plt.savefig(path_output / "figure5.png")
    plt.show()
    
def figure6():
    p = Path(path_input / "architecture_sac_perfo.csv")
    data = pd.read_csv(p)
    data["nof_training_iterations"] /= 1000000
    data["OK rising t."] *= 100
    data = data.set_index("nof_training_iterations")         
    layers = [4,3,2,1]
    neurons = [4,16,64]
    fig, axes = plt.subplots(4, 1, sharex=True, dpi=150)
    for i,layer in enumerate(layers):
        for neuron in neurons:
            df2 = data.loc[(data["nb_layers"] == layer) & (data["nb_neurons"] == neuron)]
            label = f"{neuron} neurons per layer"
            axes[i].plot(df2["OK rising t."], label=label)
            axes[i].grid()
            if i == 3:
                axes[i].set_ylabel(f"{layer} layer\n(%)")
            else:
                axes[i].set_ylabel(f"{layer} layers\n(%)") 
    box = axes[3].get_position()
    axes[3].set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.7),
              fancybox=True, shadow=True, ncol=3)
    plt.xlabel("number of training iterations (in millions)")
    plt.savefig(path_output / "figure6.png", bbox_inches="tight")
    plt.show()
