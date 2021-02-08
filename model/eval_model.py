import argparse
import os
import logging
import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1.logging as tfl  # pylint: disable=import-error
import scipy.stats
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG, PPO2, TRPO

from model.plot import display_trajectory, animate_trajectory
from utils.in_notebook import is_in_notebook
from datetime import datetime as dt 


base_dir = "/content/" if is_in_notebook() else ""


def mean_confidence_interval(data, confidence=0.95):
    if len(data) == 0: return "Empty"
    if len(data) == 1: return data[0], 0
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def f_eval(env, model):
    """Evaluate the given model on the given environment 
    """
    time_start = dt.now()
    logging.debug("f_eval() Start")
    obs = env.reset()
    y = [np.copy(env.quadcopter.state)]
    logging.debug(f"Initial Quadcopter State = {env.f_state_2_str()}")
    t = [0.0]
    cmd = [np.zeros(4)]
    w = []
    q = env.quadcopter
    r = 0
    first = [True, True, True]
    initial_error = env.query - env.quadcopter.state[7:]
    success = np.array([False, False, False])
    peak = np.zeros(3)
    failure = np.zeros(3)
    success = np.array([False, False, False])
    rise_start = np.zeros(3)
    rise_end = np.zeros(3)

    # Intent 
    # - evaluate the control on the given environment until an ending condition is reached and collect statistics about the reward, failures, success, ... 
    # 
    # Details 
    # - get action from prediction 
    # - apply the action to the environment dynamical model and get the resulting 
    #  - observation, including the error which is part of it 
    #  - reward 
    #  - ending condition 
    # - update the error normalized on the initial one 
    # - cumulate the reward 
    # - clip the action 
    # - exit only when the ending condition provided by the env is reached 
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        w.append(q.w)
        error = env.query - env.quadcopter.state[7:]
        relative_error = error / initial_error
        r += reward
        action = np.clip(action, -1., 1.)
        if done:
            break
        for i in range(3):
            if rise_start[i] == 0. and relative_error[i] < 0.9:
                rise_start[i] = env.quadcopter.t

            if rise_start[i] != 0.0 and rise_end[i] == 0.0 and relative_error[i] < 0.1:
                rise_end[i] = env.quadcopter.t

            if abs(relative_error[i]) < 0.1 and failure[i] == 0.:
                success[i] = True
            elif env.quadcopter.t > 0.5 and first[i]:
                first[i] = False
                failure[i] = np.abs(relative_error[i])
                success[i] = False

        peak = np.maximum(peak, -relative_error + 1)

        y.append(np.copy(q.state))
        t.append(q.t)
        cmd.append(np.copy(q.commands))
    y, cmd = np.array(y, dtype='float'), np.array(cmd, dtype='float')
    error = np.mean(np.abs(y[:, 6:9]), axis=0)
    time_end = dt.now()
    logging.info(f"eval() Finish --> time = {str((time_end-time_start).total_seconds())}")
    return y, w, cmd, np.array(t), failure, peak, error, success, r, rise_end - rise_start


def f_iofsw_save_trajectory(Q, y, t, w, cmd, t_cmd, checkpoint_id, iteration_time, query_memory=None, title="", plots_dir="results"):
    display_trajectory(Q=Q, y=y, t=t, w=w, cmd=cmd, t_cmd=t, query_memory=query_memory, title=f"Checkpoint {checkpoint_id} - Iteration {iteration_time}")
    temp = f"{plots_dir}/trajectory_cp_{checkpoint_id}_iteration_{iteration_time}.png"
    logging.debug(f"Operation: IO (Save), Key: Trajectory File, Value: {temp}")
    plt.savefig(temp)
    return temp 


def f_successes_list_2_success_rate(sl): 
    temp = 0 
    for e in sl: 
        if e == True: temp +=1 
    return temp/len(sl)

def mean_eval(num_episodes, model, env, plots_dir, checkpoint_id, continuous=False, v=True):
    time_start = dt.now()
    logging.debug("mean_eval() Start")
    rewards = np.zeros(num_episodes)
    success_rates = np.zeros(num_episodes)
    for i in range(num_episodes): 
        y, w, cmd, t, failure, peak, error, success, re, ris = f_eval(env=env, model=model)
        save_trajectory_args = {
            'Q' : env.quadcopter,
            'y' : y,
            'w' : w,
            't' : t,
            'cmd' : cmd,
            't_cmd' : t,
            'checkpoint_id' : checkpoint_id,
            'iteration_time' : i,
            'query_memory' : env.query_memory,
            'plots_dir' : plots_dir
        }
        f_iofsw_save_trajectory(**save_trajectory_args)
        rewards[i] = re 
        success_rates[i] = f_successes_list_2_success_rate(success)
        logging.debug(f"Evaluation Checkpoint={checkpoint_id}, Episode={i} --> Reward={rewards[i]}, Success Rate={success_rates[i]}")
    logging.debug("mean_eval() End")
    return rewards, success_rates



def f_model_2_evaluation(model, env):
    y, w, cmd, t, failure, peak, error, success, re, ris = f_eval(env=env, model=model)
    return {"env": env, "w": w, "y": y, "cmd": cmd, "t": t, "failure": failure, "peak": peak, "error": error, "success": success, "re": re, "ris": ris}


def f_iofsw_eval_2_plot(evaluation, checkpoint_id, iteration_time, plots_dir, saturated, not_saturated):
    env = evaluation['env']
    save_trajectory_args = {
        'Q' : env.quadcopter,
        'y' : evaluation['y'],
        'w' : evaluation['w'],
        't' : evaluation['t'],
        'cmd' : evaluation['cmd'],
        't_cmd' : evaluation['t'],
        'checkpoint_id' : checkpoint_id,
        'iteration_time' : iteration_time,
        'query_memory' : env.query_memory,
        'plots_dir' : plots_dir,
    }
    f_iofsw_save_trajectory(**save_trajectory_args)
    if evaluation['env'].is_pwm_1:
        plt.figure('2', figsize=(30,30))
        plt.clf()
        total = saturated + not_saturated
        plt.bar(["nominal", "saturated"],[(not_saturated / total) * 100, (saturated / total) * 100])
        plt.title(f"Proportion of nominal vs saturation cases until checkpoint {checkpoint_id}")
        plt.xlabel("mode")
        plt.ylabel("proportion (%)")
        plt.legend()
        plt.savefig(f"{plots_dir}/saturated_modes_proportion_cp_{checkpoint_id}_iteration_{iteration_time}.png")

def evaluate(model, num_steps=1000):
    env = gym.make('gym_quadcopter:quadcopter-v'+ str(args.env))
    env = DummyVecEnv([lambda: env])
    env.env_method("set_dt", dt_commands=0.01, dt=0.01)
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for _ in range(num_steps):
        actions, _states = model.predict(obs)
        obs, rewards, dones, _ = env.step(actions)

    # Stats
        for i in range(env.num_envs):
            episode_rewards[i][-1] += rewards[i]
            if dones[i]:
                episode_rewards[i].append(0.0)

    mean_rewards = [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])
        n_episodes += len(episode_rewards[i])

    # Compute mean reward
    mean_reward = round(np.mean(mean_rewards), 1)
    print(f"Mean reward: {mean_reward}, Num episodes: {n_episodes}")

    return mean_reward
