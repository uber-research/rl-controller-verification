import tensorflow.compat.v1.logging as tfl # pylint: disable=import-error
import tensorflow as tf
import gym
import numpy as np
import os
import argparse
from stable_baselines.ddpg.policies import MlpPolicy
import stable_baselines.ddpg.policies as ddpg_policies
from stable_baselines.td3.policies import MlpPolicy as td3_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
import stable_baselines.common.policies as common
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG, PPO2, TRPO, SAC, TD3
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt 
import json
import logging

from typing import Dict

from functools import partial
from google.protobuf import json_format

from model.eval_model import mean_eval, f_model_2_evaluation, f_iofsw_eval_2_plot
from utils.params_nn import get_stable_baseline_file_params
from utils.gym_gen import f_fwgym_get_env, f_fwgym_get_action_noise, f_supports_n_envs
from utils.tf_gen import f_fwtf_get_feed_dict
from utils.gen import ModelDict, EnvDict, TrainingParamsDict, SwitchParamsDict
import architectures.export

from gym_quadcopter.envs.quadcopter_2 import Quadcopter

activations = dict()
activations["relu"] = tf.nn.relu



from stable_baselines.ddpg.policies import FeedForwardPolicy

def apply_tanh_patch():
    """ Runtime patch of the Stable Baseline Code replacing the tanh activation with a clip_by_value activation
    """
    # Copied from stable_baselines github
    # https://github.com/hill-a/stable-baselines/blob/8ceda3b3b20759ab0a0e802d6428413681c57aee/stable_baselines/ddpg/policies.py#L128
    # + Patch
    def make_actor_patched(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            cnn = (self.feature_extraction == "cnn")
            pi_h = self.cnn_extractor(obs, **self.cnn_kwargs) if cnn else tf.layers.flatten(obs)
            for i, layer_size in enumerate(self.layers):
                pi_h = tf.layers.dense(pi_h, layer_size, name=f'fc{i}')
                if self.layer_norm:
                    pi_h = tf.contrib.layers.layer_norm(pi_h, center=True, scale=True)
                pi_h = self.activ(pi_h)
            # Patch is here
            self.policy = tf.clip_by_value(
                tf.layers.dense(
                    pi_h, self.ac_space.shape[0], name=scope,
                    kernel_initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
                ), -1, +1
            )
        return self.policy

    # Runtime application of the patch
    FeedForwardPolicy.make_actor = make_actor_patched


def fse_clw_get_test_env(env_id): 
    return gym.make('gym_quadcopter:quadcopter-v' + str(env_id))

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

class Training: 

    def __init__(self): 
        self.best_mean_reward = 0 
        self.n_steps =0 
        self.stats = {"rewards": []} 
        self.i =   0

    def process_end_of_actor_activation(self):
        """ Applies runtime patches to the Stable Baselines source code in order to set the End of Actor Activation
        """
        supported_values = ["tanh", "cbv"]
        if self.args.activation_end_of_actor not in supported_values:
            raise RuntimeError(f"End of Actor Activation {self.args.activation_end_of_actor} not supported")
        if self.args.activation_end_of_actor == "cbv":
            apply_tanh_patch()

    def f_clw_set_interval(self, x): 
        """ Sets the interval related to which the checkpoints are saved  
        """
        logging.debug(f"Operation: SET, Key: self.interval, Value: {x}")
        self.interval = x

    def f_clr_get_interval(self): 
        """ Gets the interval related to which the checkpoints are saved  
        """
        return self.interval

    def f_clw_set_model(self, x): 
        """ Sets the model that is used for the training   
        """
        logging.debug(f"Operation: SET, Key: self.model, Value: {x['model_name']}")
        self.model = x['model']
        self.model_name = x['model_name']

    def f_clr_get_model(self): 
        """ Gets the model that is used for the training   
        """
        logging.debug(f"Operation: GET, Key: self.model, Value: {self.model_name}") 
        return self.model 

    def f_clr_get_feed_dict(self, model): 
        feed_dict = {model.actions: model.stats_sample['actions']}

        for placeholder in [model.action_train_ph, model.action_target, model.action_adapt_noise, model.action_noise_ph]:
            if placeholder is not None:
                feed_dict[placeholder] = model.stats_sample['actions']

        for placeholder in [model.obs_train, model.obs_target, model.obs_adapt_noise, model.obs_noise]:
            if placeholder is not None:
                feed_dict[placeholder] = model.stats_sample['obs']

        return feed_dict


    def f_cb_check_switch(self): 
        if self.sp_desc.get_is_switch_active() and not self.has_switched_training_mode and (self.n_steps / self.args.n_steps) > self.sp_desc.get_time_perc(): 
            if self.sp_desc.get_is_continuous(): 
                temp = "Continuous"
                for x in self.__envs_training: 
                    x.set_continuous(quadcopter=Quadcopter(T=self.tp_desc.qg_continuous.get_T_episode(), dt_commands=self.tp_desc.qg_continuous.get_dt_command(), dt=self.tp_desc.qg_continuous.get_dt()))
            else: 
                temp = "Episodic"
                for x in self.__envs_training: 
                    x.set_episodic(quadcopter=Quadcopter(T=self.tp_desc.qg_episodic.get_T_episode(), dt_commands=self.tp_desc.qg_episodic.get_dt_command(), dt=self.tp_desc.qg_episodic.get_dt()))
            logging.info(f"QUERY MODE GENERATION SWITCH HAPPENED, now it is {temp}")
            self.has_switched_training_mode = True

    def callback(self, _locals, _globals):
        self._debug_callback(model=_locals['self'], sim_time=self.i)
        self._callback_tf_log()
        if (self.n_steps + 1) % self.f_clr_get_interval() == 0:
            self.f_cb_check_switch()
            self.i += 1
            full_checkpoint_id = int(self.model_desc.get_checkpoint_id())+int(self.i)
            logging.info(f"Checkpoint ID: Internal={self.i}, Full={full_checkpoint_id}, n_timesteps: {self.n_steps}")
            temp=self._save_model_stable_baselines(model=_locals['self'], cp_id=full_checkpoint_id)
            self._save_model_sherlock(temp)

            if self.train_saver is not None: 
                self.train_saver.save(sess=self.model.sess, save_path=f"{self.args.log_dir_tensorboard}/cp", global_step=self.i)
            if(self.args.save_as_tf): 
                path_save_cp = os.path.join(self.args.log_dir_tensorboard, f"cp-{self.i}")
                print(f"Saving Tensorflow Checkpoint in {path_save_cp}")
                self._save_model(path_save_cp)

            evaluation = f_model_2_evaluation(model=_locals['self'], env=self.env_test)
            quadcopter = self.__envs_training[0].quadcopter
            temp_plot_fn = f_iofsw_eval_2_plot(
                evaluation=evaluation, checkpoint_id=full_checkpoint_id,
                iteration_time=0, plots_dir=self.args.plots_dir,
                saturated=quadcopter.saturated, not_saturated=quadcopter.not_saturated)
            self.stats['rewards'].append(evaluation['re'])

        self.n_steps += 1
        # Returning False will stop training early
        return True

    def _debug_callback(self, model, sim_time): 
        if(self.args.debug_is_active): 
            if(self.args.debug_model_describe): 
                print(self._describe_model())
            if(self.args.debug_try_save_all_vars): 
                tf_path = f"{self.args.models_dir}/tf_quadcopter-{self.i}-desc"
                if not os.path.exists(tf_path): os.mkdir(tf_path)
                tf_testname_model = "debug_vars_all.json"
                tf_full_path = tf_path + "/" + tf_testname_model
                res = ""
                for v in tf.get_default_graph().as_graph_def().node: 
                    res += f"{v.name}\n"
                print(f"Trying to save debug data in {tf_full_path}")
                with open(tf_full_path, "w") as f: 
                    f.write(self._describe_model())
            if(self.args.debug_try_save_trainable_vars): 
                tf_path = f"{self.args.models_dir}/tf_quadcopter-{self.i}-desc"
                if not os.path.exists(tf_path): os.mkdir(tf_path)
                tf_testname_model = "debug_vars_trainable.json"
                tf_full_path = tf_path + "/" + tf_testname_model
                res = ""
                for v in tf.trainable_variables(): 
                    res += f"{v.name}\n"
                print(f"Trying to save debug data in {tf_full_path}")
                with open(tf_full_path, "w") as f: 
                    f.write(self._describe_model())
            if(self.args.debug_try_save_graph): 
                tf_path = f"{self.args.models_dir}/tf_quadcopter-{self.i}-desc"
                if not os.path.exists(tf_path): os.mkdir(tf_path)
                tf_testname_model = "debug_graph.json"
                tf_full_path = tf_path + "/" + tf_testname_model
                graph = tf.get_default_graph().as_graph_def()
                json_graph = json_format.MessageToJson(graph)
                print(f"Trying to save debug data in {tf_full_path}")
                with open(tf_full_path, "w") as f: 
                    f.write(json_graph)
            if(self.args.debug_try_save_weights): 
                tf_path = f"{self.args.models_dir}/tf_quadcopter-{self.i}-desc"
                if not os.path.exists(tf_path): os.mkdir(tf_path)
                tf_testname_model = "debug_weights.json"
                tf_full_path = tf_path + "/" + tf_testname_model
                weights = tf.trainable_variables()
                weights_vals = tf.get_default_session().run(weights)
                print(dir(tf.get_default_session().graph))
                print(f"Trying to save debug data in {tf_full_path}")
                with open(tf_full_path, "w") as f: 
                    f.write(str(weights_vals))

            if self.args.debug_show_tensors_active: 
                ops = []
                for e in self.args.debug_show_tensors_list: 
                    temp = getattr(model, e)
                    ops.append(temp)
                values = model.sess.run(ops, feed_dict=f_fwtf_get_feed_dict(model))
                for v in values: 
                    print(f"v.shape = {v.shape}\nv.value={v}\n\n")





    
    def _save_model(self, export_dir): 
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir) 
        builder.add_meta_graph_and_variables(self.model.sess, [tf.saved_model.tag_constants.TRAINING])
        builder.save()

    def _save_model_stable_baselines(self, model, cp_id): 
        # Evaluate policy training performance
        path = f"{self.args.models_dir}/quadcopter-{cp_id}{self.args.suffix}"
        logging.info(f"SAVING CURRENT MODEL, Model SAVED at {path}")
        model.save(path)
        return path + '.pkl'

    def _save_model_sherlock(self, filename): 
        output_filename = filename + '.sherlock'
        params = get_stable_baseline_file_params(filename)
        print(f"Saving Sherlock Format File {output_filename}")
        with open( output_filename, 'w' ) as file_ : 
            file_.write(architectures.export.get_sherlock_format(model_desc=self.model_desc, params=params))

    def _describe_model(self): 
        res = f"Model.Graph Type={type(self.model.graph)}\nContent={dir(self.model.graph)}\n\n\n"
        res += f"Analysing {len(tf.get_default_graph().as_graph_def().node)} nodes \n"
        res += f"Graph Def = {tf.get_default_graph().as_graph_def()}\n"
        res += f"---------\n"
        for v in tf.get_default_graph().as_graph_def().node: 
            res += f"{v.name}\n"
        res += f"-----------\n"
        return res

    def _get_action_noise(self, noise_dict, n_actions): 
        if noise_dict['name'] == 'OrnsteinUhlenbeck': 
            return OrnsteinUhlenbeckActionNoise(mean=float(noise_dict['mu'])*np.ones(n_actions), sigma=float(noise_dict['sigma']) * np.ones(n_actions))
        else: 
            raise RuntimeError(f"Unrecognized Noise Model {noise_dict['name']}")


    def _args2str(self,a): 
        return f"step={a.step}\n" \
               f"env={a.env}\n" \
               f"verbose={str(a.verbose)}\n" \
               f"save_plots={str(a.save_plots)}\n" \
               f"suffix={a.suffix}\n" \
               f"model={json.dumps(a.model)}\n" \
               f"activation={a.activation}\n" \
               f"action_noise={json.dumps(a.action_noise)}\n" \
               f"n_steps={a.n_steps}\n" \
               f"model_dir={a.models_dir}\n" \
               f"plots_dir={a.plots_dir}\n"

    def _get_plot_rewards(self): 
        fig=plt.figure("Rewards")
        plt.plot(self.stats["rewards"])
        fig.suptitle('Reward')
        plt.xlabel('time')
        plt.ylabel('reward')
        return plt

    def _write_graph_def_for_tb(self, graph_def, LOGDIR): 
        """ TODO: Remove 
        """
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(graph_def)
        train_writer.flush()
        train_writer.close()


    @property
    def sb_tb_log_active(self):
        """ Returns if native Stable Baseline Logging is active
        """
        return self.args.logging['tensorflow']['stable_baselines_native']['active']

    @property
    def sb_tb_log_dir(self):
        """ Returns the Stable Baseline TF Log Dir 
        """
        return self.args.log_dir_tensorboard if self.sb_tb_log_active else None


    def f_clr_instantiate_model(self, m): 
        res_model = None
        model_name = m.get_model_name()
        if m.get_actor_feature_extractor_type() == 'standard':
            pk = dict(act_fun=activations[self.args.activation])
        else:
            pk = dict(act_fun=activations[self.args.activation], layers=m.get_actor_feature_extractor_architecture())
        model_params = {
            'policy': MlpPolicy,
            'env': self.env,
            'verbose': int(self.args.verbose),
            'policy_kwargs': pk,
            'tensorboard_log': self.sb_tb_log_dir,
            'full_tensorboard_log': self.sb_tb_log_active
        }
        if m.get_actor_feature_extractor_name() != 'mlp':
            raise NotImplementedError(f"Exporting Policy Type {model_desc.get_actor_feature_extractor_name()} is unsupported at the moment")
        if model_name == 'ddpg':
            algo = DDPG
            model_params['param_noise'] = self.param_noise
            model_params['action_noise'] = self.action_noise
            model_params['render_eval'] = True
            model_params['policy'] = ddpg_policies.MlpPolicy
        elif model_name == 'trpo':
            algo = TRPO
            model_params['policy'] = common.MlpPolicy
        elif model_name == 'ppo':
            algo = PPO2
            model_params['policy'] = common.MlpPolicy
        elif model_name == 'td3':
            algo = TD3
            model_params['policy'] = td3_MlpPolicy
        elif model_name == 'sac':
            algo = SAC
            model_params['policy'] = sac_MlpPolicy
        model = algo(**model_params)
        # Tensorboard #
        tf.io.write_graph(model.graph, self.args.log_dir_tensorboard, "model.pbtxt")
        if self.train_writer is not None: self.train_writer.add_graph(model.graph)
        if self.train_writer is not None: self.train_writer.flush()
        logging.info(f"Instantiated Model Name={res_model}, policy={type(model_params['policy'])}, pk={pk}")
        return {"model": model, "model_name": model_name.upper()}

    def f_clw_instantiate_envs(self): 
        """ Instantiate both the Training and Test Gym Env 
        - They provide the same dynamical model and the same reward 
        """
        temp = 'gym_quadcopter:quadcopter-v' + str(self.env_desc.get_env_id())
        # TODO FIXME: Some models cannot handle multiple envs.
        N = self.env_desc.get_n_envs()
        if N < 1: 
            raise RuntimeError(f"Got NumEnvs needs to be >=1 but got NumEnvs={N}")
        logging.info(f"[SETUP] Creating {N} Training Environments - START")

        # Instantiating all the Envs and storing them into a private var 
        self.__envs_training = [f_fwgym_get_env(
            env_id=temp, used_states=self.used_states, instance_index=i,
            query_classes=self.query_classes, query_class=self.query_class,
            params=self.args.training_params
        ) for i in range(N)]

        # Passing references to previously created envs 
        self.env = DummyVecEnv([lambda: self.__envs_training[i] for i in range(N)]) 
        logging.info(f"[SETUP] Creating {N} Training Environments - DONE")
        logging.info(f"[SETUP] Creating 1 Test Environments - START")
        self.env_test = f_fwgym_get_env(
            env_id=temp, used_states=self.used_states, instance_index=0,
            query_classes=self.query_classes, query_class=self.query_class,
            params=self.args.testing_params
        )
        logging.info(f"[SETUP] Creating 1 Test Environments - DONE")

    def f_clw_args_2_state(self, args): 
        """Initialize internal instance state 
        """
        self.model_desc = ModelDict(model_dict=self.args.model)
        self.env_desc = EnvDict(env_dict=self.args.env)
        self.tp_desc = TrainingParamsDict(tp_dict=self.args.training_params)
        self.sp_desc = SwitchParamsDict(self.tp_desc.get_switch_params()) 
        self.query_classes = self.args.query_classes
        self.query_class = self.args.query_class
        self.used_states = self.args.used_states
        self.train_writer = None
        self.param_noise = None

        self.f_clw_instantiate_envs()
        self.n_actions = self.env.action_space.shape[-1]

        self.action_noise = f_fwgym_get_action_noise(noise_dict=self.args.action_noise, n_actions=self.n_actions)
        self.has_switched_training_mode = False


    def f_fwtfw_init(self): 
        """Initialize TF Environment 
        """
        tfl.set_verbosity(tfl.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    def get_global_summary(self): 
        return {"ModelName": self.model_desc.get_model_name(), "Continuous": str(self.tp_desc.get_is_continuous()), "Total_Training_Iterations": self.args.n_steps, "Iterations_Per_Checkpoint": self.args.iterations_checkpoint, "Env" : { "ID" : self.env_desc.get_env_id(), "Num_Envs" : self.env_desc.get_n_envs() }}

    def _add_tf_logs(self):
        """Adds the additional Tensorflow Logs to the standard Stable Baselines ones
        """
        with self.model.graph.as_default():
            # Conditional Logging for Summary
            if self.args.logging['tensorflow']['summary']['active']:
                tf.summary.text('Env Summary', tf.convert_to_tensor(str(self.env)))
            
            # Conditional Logging for the Stable Baselines Tensors specified in the list
            if self.args.logging['tensorflow']['stable_baselines_tensors']['active']:
                for e in self.args.logging['tensorflow']['stable_baselines_tensors']['list']:
                    tf.summary.scalar(f"Custom_SB_Log/{e}", tf.reduce_mean(getattr(self.model, e)))

            # Conditional Logging for the Tensorflow Tensors specified in the list
            if self.args.logging['tensorflow']['tensorflow_tensors']['active']:
                for e in self.args.logging['tensorflow']['tensorflow_tensors']['list']:
                    tf.summary.histogram(f"Custom_TF_Log/{e}", tf.get_default_graph().get_tensor_by_name(e))

            # Conditional Logging for Quadcopter Framework Events
            if self.args.logging['tensorflow']['events']['active']:
                if 'on_step' in self.args.logging['tensorflow']['events']['list']:
                    tf.summary.text(f'EnvStep{self.n_steps}', tf.convert_to_tensor(self.env.env_method('get_on_step_log')))

            # Merge all of the added summaries 
            self.model.summary = tf.summary.merge_all()


    def _callback_tf_log(self):
        with self.model.graph.as_default():
            if self.args.logging['tensorflow']['events']['active']:
                if 'on_step' in self.args.logging['tensorflow']['events']['list']:
                    tf.summary.text('EnvStep', tf.convert_to_tensor(self.env.env_method('get_on_step_log')))
                    self.model.summary = tf.summary.merge_all()

    def run_training(self, args):
        """ Training Function
        """
        # Use standard log just for the initial setup
        # Set the log used during training
        self.args = args
        self.process_end_of_actor_activation()
        self.f_clw_args_2_state(args)
        logging.info(f"Train Arguments\n{self._args2str(self.args)}") 
        logging.info(f"Writing Tensorboard Log to {self.args.log_dir_tensorboard}")
        logging.info(f"Start training at {dt.now().strftime('%Y%m%d_%H%M')}")
        self.f_fwtfw_init()
        if self.model_desc.get_is_load():
            # TODO: Fix this part 
            path = self.model_desc.get_checkpoint_path()
            model_name = self.model_desc.get_model_name()
            logging.info(f"LOADING MODEL at {path}")
            if model_name == "ddpg":
                self.model = DDPG.load(path, self.env)
            elif model_name == "ppo":
                self.model = PPO2.load(path, self.env)
            elif model_name == "trpo":
                self.model = TRPO.load(path, self.env)
            elif model_name == "td3":
                self.model = TD3.load(path, self.env)
            elif model_name == 'sac':
                self.model = SAC.load(path, self.env)
        else:
            # the noise objects for DDPG
            self.f_clw_set_model(self.f_clr_instantiate_model(m=self.model_desc))

            
        self.f_clw_set_interval(self.args.iterations_checkpoint)

        if self.args.save_tf_checkpoint: 
            with self.model.graph.as_default(): 
                self.train_saver = tf.compat.v1.train.Saver()
        else: 
            self.train_saver = None 



        self.i = 0
        # Implemented in 
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ddpg/ddpg.py#L807

        logging.info(f"GLOBAL SUMMARY: {self.get_global_summary()}")
        self._add_tf_logs()
        self.model.learn(total_timesteps=int(self.args.n_steps), callback=self.callback)
        logging.info(f"Training Finished after {self.n_steps} iterations saving {self.i} intermediate checkpoints")
        logging.info(f"Saving Final Model in Stable Baseline Checkpoint")
        temp=self._save_model_stable_baselines(model=self.model, cp_id="final")

        print(f"Exporting Actor from Final Model in Stable Baseline Checkpoint as Sherlock Format")
        self._save_model_sherlock(temp)

        if self.train_writer is not None: self.train_writer.close()

        plt = self._get_plot_rewards()
        now = dt.now() 
        plt.savefig(f"{self.args.plots_dir}/reward_{now.strftime('%Y%m%d_%H%M%S')}.png")
        return True
