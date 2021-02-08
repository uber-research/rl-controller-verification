from typing import Tuple 
import tensorflow.compat.v1.logging as tfl # pylint: disable=import-error
import gym
import numpy as np
import os
import scipy.stats
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG, PPO2, TRPO, TD3, SAC
from model.eval_model import mean_eval, evaluate, mean_confidence_interval
from utils.gen import f_iofsw_plot
from utils.gym_gen import f_fwgym_get_env, f_fwgym_get_action_noise
import matplotlib.pyplot as plt
from datetime import datetime as dt 
import pandas as pd
import logging
from pathlib import Path

def load_model(path: str, env, desc: str):
    """ Loads a model from a stable baseline checkpoint file into a memory representation 

    Args:
        path        (str)           :       Path to the Stable Baseline Checkpoint File 
        env         (SB Env)        :       Path to the Stable Baseline Checkpoint File 
        desc        (str)           :       Text Description of what model this is

    Returns:
        The loaded model
    """

    if desc == "ddpg":
        return DDPG.load(path, env)
    elif desc == "ppo":
        env = DummyVecEnv([lambda: env])
        return PPO2.load(path, env)
    elif desc == "trpo":
        env = DummyVecEnv([lambda: env])
        return TRPO.load(path, env)
    elif desc == "td3":
            return TD3.load(path, env)
    elif desc == "sac":
            return SAC.load(path, env)
    else:
        raise RuntimeError(f"Model Name {desc} not supported")


def get_checkpoint_path(base_path: str, idx: int, suffix: str):
    """ Returns the checkpoint path given the necessary information to compute it 

    Args:
        base_path       (str)       :       The Experiment Dir Path
        idx             (int)       :       The Checkpoint Index
        suffix          (str)       :       The checkpoint file suffix

    Returns:
        The checkpoint full path
    """
    return f"{base_path}/models/quadcopter-{idx}{suffix}"


def get_signals_path(basepath_checkpoint: str, episode_idx: int):
    """ Builds the full path and filename for each signal

    Args:
        basepath_checkpoint         (str)       :       The container dir for the 3 result files
        episode_idx                 (int)       :       The episode index  

    """
    base_fn = f"{basepath_checkpoint}" + f"/episode_{episode_idx}"
    # This file contains a sequence of [1..N] queries concatenated
    # File header: t,query_p,query_q,query_r
    query_fp = f"{base_fn}.queries.csv"

    # This file contains the sequence of commands sent to reach the query 
    # File header: t_cmd,cmd_thrust,cmd_phi,cmd_theta,cmd_psi
    commands_fp = f"{base_fn}.commands.csv"

    # This file contains the sequence of states reached while trying to reach the query
    # File header: t,z,u,v,w,phi,theta,psi,p,q,r
    signals_fp = f"{base_fn}.signals.csv"
    return query_fp, commands_fp, signals_fp

def get_signals_containers():
    """ Builds the data structures for each signal which will be then converted into CSV with the right header
    """
    query = pd.DataFrame(columns=["t","query_p","query_q","query_r"])
    cmds = pd.DataFrame(columns=["t_cmd","cmd_thrust","cmd_phi","cmd_theta","cmd_psi"])
    signals = pd.DataFrame(columns=["t","z","u","v","w","phi","theta","psi","p","q","r","w_1","w_2","w_3","w_4"])
    return query, cmds, signals

class Test:
    """Runs Tests for a range of checkpoints on a given environment.

    Attributes:
        env : Test Environment

    """
    def __init__(self, args):
        self.args = args
        self.base_dir = ""


    def _args2str(self,a):
        return f"step={a.step}\n" + \
               f"env={a.env}\n" + \
               f"suffix={a.suffix}\n" + \
               f"n_episodes={a.n_episodes}\n" + \
               f"model={a.model}\n" + \
               f"continuous={a.continuous}\n" + \
               f"save_plots={a.save_plots}\n" + \
               f"start_index={a.start_index}\n" + \
               f"end_index={a.end_index}\n" + \
               f"plots_dir={a.plots_dir}"



    def f_checkpoints_range_2_mean_performance(self, checkpoints: range) -> Tuple[np.ndarray, np.ndarray]: 
        logging.debug(f"[f_checkpoints_range_2_mean_performance]: checkpoints={checkpoints}")
        rewards = np.zeros(len(checkpoints))
        s_rates = np.zeros(len(checkpoints))
        # Intent 
        # - Iterate over this range, to load the associated Stable Baseline Model Checkpoint 
        # - Pass that model to `mean_eval` evaluation function which will evaluate the model on 
        #   - a certain number of episodes 
        #   - a certain env 
        #    - continuous or not continuous space 
        # - an evaluation returns reward and average success rate 
        # 
        # Evaluating N checkpoints on M queries and then averaging on M so to finally have N Rewards and N Success Rates 

        j = 0 
        """ NOTE: i can range in anyway while j iterates over the numpy array 
        """
        for i in checkpoints:
            path = f"{self.args.training_base_path}/models/quadcopter-{i}{self.args.suffix}"
            logging.debug(f"Evaluating model at {path}")
            if self.args.model['name'] == "ddpg":
                model = DDPG.load(path)
            elif self.args.model['name'] == "ppo":
                model = PPO2.load(path)
            elif self.args.model['name'] == "trpo":
                model = TRPO.load(path)
            elif self.args.model['name'] == "td3":
                model = TD3.load(path)
            elif self.args.model['name'] == "sac":
                model = SAC.load(path)
            logging.debug(f"Evaluating Model {self.args.model['name']} for {self.args.n_episodes} episodes in {self.args.env} environment with continuous={str(self.args.continuous)}")
            rewards_list, success_rates_list = mean_eval(num_episodes=self.args.n_episodes, checkpoint_id=i, model=model, env=self.env, v=True, continuous=self.args.continuous, plots_dir=self.args.plots_dir)
            rewards_mean = np.mean(rewards_list)
            success_rates_mean = np.mean(success_rates_list)
            logging.debug(f"Evaluation Checkpoint={i} --> Average Reward = {rewards_mean}, Average Success Rate = {success_rates_mean}")
            rewards[j] = rewards_mean
            s_rates[j] = success_rates_mean
            j += 1
        return rewards, s_rates

    def f_fwtfw_init(self): 
        """Initialize TF Environment 
        """
        tfl.set_verbosity(tfl.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def f_iofsw_plot_rewards(self, rewards, y_min=-3, y_max=0): 
        """ Plots a zero centered plot of the reward as a function of the training time 
        """
        plt.figure("Rewards")
        plt.plot(
            [0]+[self.args.num_iterations_checkpoint * i for i in range(self.args.start_index, self.args.end_index, self.args.step)],
            [0]+ list(rewards / np.max(np.abs(rewards)))
        )
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Reward")
        plt.yticks([-3, 0])
        path_fig = f"{self.args.plots_dir}/Timesteps_Rewards_{dt.now().strftime('%Y%m%d_%H%M%S')}.png"
        logging.info(f"Saving Figure to {path_fig}")
        plt.savefig(path_fig)



    def f_iofsw_plot_success_rate(self, s_rates): 
        logging.info(f"Sucess Rates Shape {s_rates.shape}, Val = {s_rates}")
        i_max = np.argmax(s_rates)
        logging.info(f"[RESULT]    Best average success rate: {s_rates[i_max]}, achieved by checkpoint {self.args.start_index+(i_max*self.args.step)}")

        plt.figure("Success rates")
        plt.plot(
            [0]+[self.args.num_iterations_checkpoint * i for i in range(self.args.start_index, self.args.end_index, self.args.step)],
            [0]+list(s_rates)
        )
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Success Rate")
        plt.yticks([0, 1.0])
        now = dt.now() 
        path_fig = f"{self.args.plots_dir}/SuccessRate_%s.png" % now.strftime("%Y%m%d_%H%M%S")
        logging.info(f"Saving figure to {path_fig}")
        plt.savefig(path_fig)





    def run_test(self):
        """ Runs the old tests or the new properties observer based ones
        """
        if self.args.eval_properties_observer['is_active']:
            return self.compute_traces()

        logging.info(f"Start test at {dt.now().strftime('%Y%m%d_%H%M')}")
        logging.info(f"Test Arguments\n{self._args2str(self.args)}")
        self.f_fwtfw_init()

        temp = 'gym_quadcopter:quadcopter-v' + str(self.args.env)

        self.env = f_fwgym_get_env(
            env_id=temp,
            used_states = self.args.used_states,
            instance_index=0, query_classes=self.args.query_classes,
            query_class=self.args.query_class, params=self.args.testing_params
        )
        logging.info(f"[eval_models.py] Instantiated env {str(temp)} with continuous {str(self.args.continuous)}")
        checkpoints = self.range_checkpoints()
        rewards, s_rates = self.f_checkpoints_range_2_mean_performance(checkpoints=checkpoints)

        starting_min_reward = -10
        
        temp_x = [0]+[self.args.num_iterations_checkpoint * i for i in range(self.args.start_index, self.args.end_index, self.args.step)]
        temp_y = [starting_min_reward]+ list(rewards)
        f_iofsw_plot(
            x=temp_x, 
            y=temp_y, 
            x_ticks=np.array(temp_x), 
            y_ticks=np.array(temp_y), 
            title="Rewards", 
            label_x="Training Timesteps", 
            label_y="Average Rewards", 
            filename=f"{self.args.plots_dir}/Timesteps_Rewards_{dt.now().strftime('%Y%m%d_%H%M%S')}.png", 
        )

        temp_x = [0]+[self.args.num_iterations_checkpoint * i for i in range(self.args.start_index, self.args.end_index, self.args.step)]
        temp_y = [0]+list(s_rates)
        f_iofsw_plot(
            x=temp_x, 
            y=temp_y, 
            x_ticks=np.array(temp_x), 
            y_ticks=np.array(temp_y), 
            title="Success Rate", 
            label_x="Training Timesteps", 
            label_y="Average Success Rate", 
            filename=f"{self.args.plots_dir}/Timesteps_SuccessRate_{dt.now().strftime('%Y%m%d_%H%M%S')}.png", 
        )


    def cb_quadcopter_step(self):
        w = self.env.quadcopter.w
        signals_item = {
            't': self.env.quadcopter.t.hex(),
            'z': self.env.quadcopter.state[0].hex(),
            'u': self.env.quadcopter.state[1].hex(),
            'v': self.env.quadcopter.state[2].hex(),
            'w': self.env.quadcopter.state[3].hex(),
            'phi': self.env.quadcopter.state[4].hex(),
            'theta': self.env.quadcopter.state[5].hex(),
            'psi': self.env.quadcopter.state[6].hex(),
            'p': self.env.quadcopter.state[7].hex(),
            'q': self.env.quadcopter.state[8].hex(),
            'r': self.env.quadcopter.state[9].hex(),
            'w_1': w[0].hex(),
            'w_2': w[1].hex(),
            'w_3': w[2].hex(),
            'w_4': w[3].hex()
        }
        self.signals = self.signals.append(signals_item, ignore_index = True)
        cmd_item = {
            't_cmd': self.env.quadcopter.t.hex(),
            # NOTE: To have the thrust related to this timestep it is necessary to call step() first
            'cmd_thrust': float(self.env.commands[0]).hex(),
            'cmd_phi': float(self.env.commands[1]).hex(),
            'cmd_theta': float(self.env.commands[2]).hex(),
            'cmd_psi': float(self.env.commands[3]).hex()
        }
        self.cmds = self.cmds.append(cmd_item, ignore_index=True)


    def compute_traces(self):
        """ For each Stable Baseline Checkpoin in the experiment dir, given a certain Query Generator Configuration it compute traces

        All the Stable Baseline Checkpoints are loaded and run in an Env with a properly istantiated Query Generator
        """

        # Instantiate the Env for the Tests
        env_id = 'gym_quadcopter:quadcopter-v' + str(self.args.env)
        self.env = f_fwgym_get_env(
            env_id=env_id, used_states = self.args.used_states,
            instance_index=0, query_classes=self.args.query_classes,
            query_class=self.args.query_class, params=self.args.testing_params
        )

        # Register Callback
        self.env.quadcopter.cb_step = self.cb_quadcopter_step
        logging.info(f"[eval_models.py] Instantiated env {env_id} with continuous {str(self.args.continuous)}")
        checkpoints = self.range_checkpoints()

        # Iterate over the checkpoints
        for i in checkpoints:
            # Input: Checkpoints Dir
            cp_path = get_checkpoint_path(base_path=self.args.training_base_path, idx = i, suffix = self.args.suffix)
            # Loads the model from the Checkpoint
            model = load_model(path = cp_path, env=self.env, desc = self.args.model['name'])

            # Create a dir for the traces related to a given checkpoint, using `self.args.log_dir` as base
            base_path_cp_id = self.args.log_dir + f"/checkpoint_{i}/"
            Path(base_path_cp_id).mkdir()

            # Iterate over the episodes to test each checkpoint
            for j in range(self.args.n_episodes):

                # Get the evaluation results filenames
                query_full_path, commands_full_path, signals_full_path = get_signals_path(basepath_checkpoint = base_path_cp_id, episode_idx = j)

                # Get the data structures for the evaluated data 
                self.query, self.cmds, self.signals = get_signals_containers()

                # Reset the env at the beginning of each episode
                obs = self.env.reset()

                # Simulation loop
                while True:
                    # NOTE: The time granularity here should be the dt_commands one and not the dt one as it is sync with step() method
                    query_item = {
                        't': self.env.quadcopter.t.hex(),
                        'query_p': float(self.env.query[0]).hex(), 
                        'query_q': float(self.env.query[1]).hex(), 
                        'query_r': float(self.env.query[2]).hex()
                        }
                    self.query=self.query.append(query_item, ignore_index=True)

                    # Get the action from the Actor
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.env.step(action)

                    if done:
                        break
            
                self.query.to_csv(query_full_path, index = False)
                self.cmds.to_csv(commands_full_path, index = False)
                self.signals.to_csv(signals_full_path, index = False)

    def range_checkpoints(self):
        """ Defines the Checkpoint Range to evaluate """
        p = Path(f"{self.args.training_base_path}/models/")
        max_checkpoint = int(len([x for x in p.iterdir()])/2 + 1)
        quadcopter_file = p / f"quadcopter-{max_checkpoint}.pkl.sherlock"
        while not quadcopter_file.is_file() and max_checkpoint > 0:
            max_checkpoint -= 1
            quadcopter_file = p / f"quadcopter-{max_checkpoint}.pkl.sherlock"
        end_index = min(self.args.end_index, max_checkpoint) + 1
        return range(self.args.start_index, end_index, self.args.step)
