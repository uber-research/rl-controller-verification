import tensorflow.compat.v1.logging as tfl # pylint: disable=import-error
import gym
import numpy as np
import os
import argparse
import scipy.stats
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG, PPO2, TRPO

from model.eval_model import mean_eval

from datetime import datetime
import matplotlib.pyplot as plt

class Plots: 
  def __init__(self): 
    self.base_dir = ""

  def get_standard_args(self): 
    class Namespace(object): 
        pass 
    temp = Namespace()
    temp.env = 0
    temp.n_episodes = 100
    temp.plots_dir = 'results'

    return temp 

  def args2str(self, a): 
      return f"env={a.env}, n_episodes={a.n_episodes}, plots_dir={a.plots_dir}"



  def my_compute_data(self, args, env, params, n_episodes): 
    env = gym.make('gym_quadcopter:quadcopter-v' + str(args.env))
    for alg, start_index, end_index, step, suffix in params: 
      re_d = []
      sr_d = []
      rewards, s_rates = [], []
      for i in range(start_index, end_index, step):
          print("")
          print(f"Working on alg={alg}, start_index={start_index}, end_index={end_index}, step={step}, suffix={suffix}, i={i}")
          path = f"{self.base_dir}models/{alg}/quadcopter-v{args.env}-{i}{suffix}.pkl" 
          print(f"Evaluating model at {path}")
          if not os.path.exists(path): 
            print(f"WARNING: File {path} does not exist --> SKIPPING")
            continue

          if alg == "ddpg":
              model = DDPG.load(path)
          elif alg == "ppo":
              model = PPO2.load(path)
          else:
              model = TRPO.load(path)
          r, su = mean_eval(
              n_episodes,
              model,
              env, False, False
          )
          print(f"Average Success Rate: {su}")
          rewards.append(r)
          s_rates.append(su[0])

      i_max = np.argmax(s_rates)
      re_d.append(rewards)
      sr_d.append(s_rates)
      return re_d, sr_d


  def my_plot(self, re_d, sr_d, plots_dir): 
      fig = plt.subplot(121)
      algs = ["ddpg", "trpo", "ppo"]
      intervals = [1000*50, 10000*10, 128*1000]
      norm =  - min(min(re_d))
      for i in range(len(re_d)):
        re_d[i] = np.array(re_d[i])/norm
        sr_d[i] = np.array(sr_d[i])
      for i in range(len(re_d)):
        plt.plot(
          [intervals[i]*j for j  in range(len(re_d[i]))],
            re_d[i],
            label=algs[i]
        )
      plt.xlabel("Timesteps")
      plt.ylabel("Average reward")
      plt.legend()

      fig = plt.subplot(122)
      for i in range(len(sr_d)):
          plt.plot(
            [intervals[i]*j for j  in range(len(sr_d[i]))],
            sr_d[i],
            label=algs[i]
          )
      plt.xlabel("Timesteps")
      plt.ylabel("Average Success Rate")
      plt.legend()
      
      path_fig = f"{self.base_dir}{plots_dir}/Timesteps_AverageSuccessRate{datetime.now().strftime('plot_%Y%m%d_%H_%M_%S')}.png"
      print(f"Saving Figure {path_fig}")
      plt.savefig(path_fig)


  def run_plots(self, args): 
      tfl.set_verbosity(tfl.ERROR)
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
      print(f"Test Arguments\n{self.args2str(args)}")

      params = [("ddpg", 1, 1720, 20, ""), ("trpo", 1, 292, 10, ""), ("ppo", 1, 7, 1, "final_test")]

      re_d, sr_d = self.my_compute_data(args=args, env=args.env, params=params, n_episodes=args.n_episodes)
      self.my_plot(re_d=re_d, sr_d=sr_d, plots_dir=args.plots_dir)
