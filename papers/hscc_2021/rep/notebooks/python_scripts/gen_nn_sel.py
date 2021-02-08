#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:35:08 2020

@author: arthur.gold
"""

from pathlib import Path
import pandas as pd
import numpy as np

p_df = Path("mean_aggregated_all.csv")
path_folder = Path("network_selections")
df = pd.read_csv(p_df)
df = df.rename(columns={'Unnamed: 0': 'uid'})
df.insert(1,"from_uid", np.nan)
df = df.reset_index()
df.drop("index", axis=1, inplace=True)
df.set_index('uid', inplace=True)
sac = (df["algo"]=="sac") & (df["pid_rates"]=="None")
ddpg = (df["algo"]=="ddpg") & (df["pid_rates"]=="None")
ppo = (df["algo"]=="ppo") & (df["pid_rates"]=="None")
td3 = (df["algo"]=="td3") & (df["pid_rates"]=="None")
pid = (df["pid_rates"] != "None")
three_states = (df["p"] == False) & (df["thrust"] == False)
six_states = (df["p"] == True) & (df["thrust"] == False)
seven_states = (df["p"] == True) & (df["thrust"] == True)
nominal = (df["training_windgust_magnitude_max"] == 1) & (df["test_windgust_magnitude_max"] == 1) \
& (df["training_saturation_motor"] == 1) & (df["test_saturation_motor"] == 1)
windgust_nominal = (df["training_windgust_magnitude_max"] == 10) & (df["test_windgust_magnitude_max"] == 1)
windgust_windgust = (df["training_windgust_magnitude_max"] == 10) & (df["test_windgust_magnitude_max"] == 10)
nominal_windgust = (df["training_windgust_magnitude_max"] == 1) & (df["test_windgust_magnitude_max"] == 10)
saturation_saturation = (df["training_saturation_motor"] == 0.8) & (df["test_saturation_motor"] == 0.8)
saturation_nominal = (df["training_saturation_motor"] == 0.8) & (df["test_saturation_motor"] == 1)
nominal_saturation = (df["training_saturation_motor"] == 1) & (df["test_saturation_motor"] == 0.8)

def architecture_sac_perfo():
    sac = (df["algo"]=="sac") & (df["pid_rates"]=="None")
    three_states = (df["p"] == False) & (df["thrust"] == False)
    nominal = (df["training_windgust_magnitude_max"] == 1) & (df["test_windgust_magnitude_max"] == 1) \
    & (df["training_saturation_motor"] == 1) & (df["test_saturation_motor"] == 1)
    data = df.loc[nominal & three_states & sac]
    data["uid"] = 0
    data["from_uid"] = "null"
    data = df.loc[nominal & three_states & sac]
    data.to_csv(path_folder / "architecture_sac_perfo.csv")

def ddpg_best():
    data = df.loc[ddpg & nominal & (df["OK rising t."]>0.9904) & (df["OK off."]>0.9858) & (df["OK overshoot"] > 0.9820)]
    data.to_csv(path_folder / "ddpg_best.csv")

def ppo_best():
    data = df.loc[ppo & nominal & (df["OK rising t."]>0.965) & (df["OK off."]>0.97) & (df["OK overshoot"] > 0.91)]
    data.to_csv(path_folder / "ppo_best.csv")
  
def td3_best():
    data = df.loc[td3 & nominal & (df["OK rising t."]>0.96) & (df["OK off."]>0.86) & (df["OK overshoot"] > 0.88)]
    data.to_csv(path_folder / "td3_best.csv")

def sac_best():
    data = df.loc[sac & nominal & (df["OK rising t."]>0.97) & (df["OK off."]>0.99) & (df["OK overshoot"] > 0.96)]
    data.to_csv(path_folder / "sac_best.csv")
    
def pid_best():
    data = df.loc[pid & nominal]
    data.to_csv(path_folder / "pid_best.csv")

def ddpg_sac_saturation_test_saturation():
    data = df.loc[saturation_saturation & (df["OK rising t."]>0.89) & (df["OK off."]>0.7) & (df["OK overshoot"]>0.9)]
    data.to_csv(path_folder / "ddpg_sac_saturation_test_saturation.csv")
    
def ddpg_sac_windgust_test_nominal():
    data = df.loc[windgust_nominal & (df["OK rising t."]>0.95) & (df["OK off."]>0.91) & (df["OK overshoot"]>0.94)]
    data.to_csv(path_folder / "ddpg_sac_windgust_test_nominal.csv")

def ddpg_sac_windgust_test_windgust():
    data = df.loc[windgust_windgust & (df["OK rising t."]>0.975) & (df["OK off."]>0.89) & (df["OK overshoot"]>0.92)]
    data.to_csv(path_folder / "ddpg_sac_windgust_test_windgust.csv")

def ddpg_saturation_test_nominal():
    data = df.loc[saturation_nominal & ddpg & (df["OK rising t."]>0.94) & (df["OK off."]>0.95) & (df["OK overshoot"]>0.94)]
    data.to_csv(path_folder / "ddpg_saturation_test_nominal.csv")

def nominal_test_saturation():
    data = df.loc[nominal_saturation & (df["OK rising t."]>0.94) & (df["OK off."]>0.8) & (df["OK overshoot"]>0.92)]
    data.to_csv(path_folder / "nominal_test_saturation.csv")
    
def nominal_test_windgust():
    data = df.loc[nominal_windgust & (df["OK rising t."]>0.98) & (df["OK off."]>0.97) & (df["OK overshoot"]>0.984)]
    data.to_csv(path_folder / "nominal_test_windgust.csv")

def pid_test_saturation():
    data = df.loc[nominal_saturation & pid]
    data.to_csv(path_folder / "pid_test_saturation.csv")
    
def pid_test_windgust():
    data = df.loc[nominal_windgust & pid]
    data.to_csv(path_folder / "pid_test_windgust.csv")
    
def sac_3D():
    data = df.loc[sac & three_states & (df["OK rising t."]>0.98) & (df["OK off."]>0.98) & (df["OK overshoot"]>0.985)]
    data.to_csv(path_folder / "sac_3D.csv")

def sac_6D():
    data = df.loc[sac & six_states & (df["OK rising t."]>0.97) & (df["OK off."]>0.98) & (df["OK overshoot"]>0.94)]
    data.to_csv(path_folder / "sac_6D.csv")
    
def sac_7D():
    data = df.loc[sac & seven_states & (df["OK rising t."]>0.94) & (df["OK off."]>0.94) & (df["OK overshoot"]>0.97)]
    data.to_csv(path_folder / "sac_7D.csv")

def sac_32_32():
    data = df.loc[nominal & three_states & sac & (df["nb_layers"] == 2) & (df["nb_neurons"] == 32) ]
    data.to_csv(path_folder / "sac_32_32.csv")

def sac_saturation_test_nominal():
    data = df.loc[saturation_nominal & sac & (df["OK rising t."]>0.95) & (df["OK off."]>0.95) & (df["OK overshoot"]>0.97)]
    data.to_csv(path_folder / "sac_saturation_test_nominal.csv")
