#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:33:39 2020

@author: arthur.gold
"""

import pandas as pd
from pathlib import Path

path_input = Path("network_selections")
path_output = Path("generated_tables")

def table1():
    names = ["pid", "sac", "ddpg", "ppo", "td3"]
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    datas = []
    for name in names:
        p = Path(path_input / f"{name}_best.csv")
        data = pd.read_csv(p)
        if name == "pid":
            data.loc[data["pid_rates"]=="pid_rates_crazyflie",["pid_rates"]] = "pid1"
            data.loc[data["pid_rates"]=="pid_rates_better",["pid_rates"]] ="pid2"
            data["algo"] = data["pid_rates"]
        for OK in OKs:
            data[OK] *= 100
        datas.append(data)
        
    df = pd.concat(datas, axis=0)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot"]
    df = df[selected_columns]
    df.set_index("algo", inplace=True)
    print(df)
    df.to_csv(path_output / "table1.csv", float_format='%.2f')
    
def table2():
    names = ["_3D", "_6D", "_7D"]
    datas = []
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    for name in names:
        p = Path(path_input / f"sac{name}.csv")
        data = pd.read_csv(p)
        data["dim"] = name[-2]
        for OK in OKs:
            data[OK] *= 100
        datas.append(data)

    df = pd.concat(datas, axis=0)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot", "dim"]
    df = df[selected_columns]
    df.set_index(["algo", "dim"], inplace=True)
    print(df)
    df.to_csv(path_output / "table2.csv", float_format='%.2f') 
  
def table3(): 
    names_1 = ["nominal", "pid"]
    names_2 = ["windgust", "saturation"]
    datas = []
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    for name2 in names_2:
        for name1 in names_1:
            p = Path(path_input / f"{name1}_test_{name2}.csv")
            data = pd.read_csv(p)
            data["mode"] = name2
            if name1 == "pid":
                data.loc[data["pid_rates"]=="pid_rates_crazyflie",["pid_rates"]] = "pid1"
                data.loc[data["pid_rates"]=="pid_rates_better",["pid_rates"]] ="pid2"
                data["algo"] = data["pid_rates"]
            for OK in OKs:
                data[OK] *= 100
            datas.append(data)
    df = pd.concat(datas, axis=0)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot", "mode"]
    df = df[selected_columns]
    df.set_index("mode", inplace=True)
    print(df)
    df.to_csv(path_output / "table3.csv", float_format='%.2f')
    
    
def table4():
    p = Path(path_input / "ddpg_sac_saturation_test_saturation.csv")
    data = pd.read_csv(p)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot"]
    data = data[selected_columns]
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    for OK in OKs:
        data[OK] *= 100
    data.set_index("algo", inplace=True)
    print(data)
    data.to_csv(path_output / "table4.csv", float_format='%.2f')
    
def table5():
    names = ["sac", "ddpg"]
    datas = []
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    for name in names:
        p = Path(path_input / f"{name}_saturation_test_nominal.csv")
        data = pd.read_csv(p)
        for OK in OKs:
            data[OK] *= 100
        datas.append(data)
    df = pd.concat(datas, axis=0)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot"]
    df = df[selected_columns]
    df.set_index("algo", inplace=True)
    print(df)
    df.to_csv(path_output / "table5.csv", float_format='%.2f')
    
def table6():
    p = Path(path_input / "ddpg_sac_windgust_test_windgust.csv")
    data = pd.read_csv(p)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot"]
    data = data[selected_columns]
    data.set_index("algo", inplace=True)
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    for OK in OKs:
        data[OK] *= 100
    print(data)
    data.to_csv(path_output / "table6.csv", float_format='%.2f')

def table7():
    p = Path(path_input / "ddpg_sac_windgust_test_nominal.csv")
    data = pd.read_csv(p)
    selected_columns = ["algo", "OK rising t.", "OK off.", "OK overshoot", "avg rising t.", "avg off.", "avg overshoot", "max rising t.", "max off.", "max overshoot"]
    data = data[selected_columns]
    data.set_index("algo", inplace=True)
    OKs = ["OK rising t.", "OK off.", "OK overshoot"]
    for OK in OKs:
        data[OK] *= 100
    print(data)
    data.to_csv(path_output / "table7.csv", float_format='%.2f')
