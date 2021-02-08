import tensorflow.compat.v1.logging as tfl #
import gym
import tensorflow as tf
import numpy as np
import matplotlib
import os
import argparse
import scipy.stats

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG, PPO2, TRPO

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from operator import itemgetter


def write_nn(model):
    return 0

if __name__ == "__main__":
    tfl.set_verbosity(tfl.ERROR)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--step', dest='step', default=0, type=int)
    parser.add_argument('--env', dest='env', default=0, type=int)
    parser.add_argument('--suffix', dest='suffix', default='', type=str)
    parser.add_argument('--alg', dest='alg', default='ddpg', type=str)
    parser.add_argument('--output_path', dest="path", default="NN_files/", type=str)

    args = parser.parse_args()
    env = gym.make('gym_quadcopter:quadcopter-v' + str(args.env))
    print("[INFO]   Loading model at models/{}/quadcopter-v{}-{}{}".format(args.alg, args.env, args.step, args.suffix))
    print()
    if args.alg == "ddpg":
            model = DDPG.load("models/{}/quadcopter-v{}-{}{}".format(args.alg, args.env, args.step, args.suffix))
    elif args.alg == "ppo":
            model = PPO2.load("models/{}/quadcopter-v{}-{}{}".format(args.alg, args.env, args.step, args.suffix))
    elif args.alg == "trpo":
            model = TRPO.load("models/{}/quadcopter-v{}-{}{}".format(args.alg, args.env, args.step, args.suffix))

    x = model.sess.run(itemgetter(0, 2, 4)(model.params))
    b = model.sess.run(itemgetter(1, 3, 5)(model.params))
    print(x[2])
    print(b[2])

    filename = "{}/{}/{}-{}".format(args.path, args.alg, args.suffix, args.step)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, "w") as f:
        f.write("14\n3\n2\n64\n64\n")
        for k in range(3):
            for i in range(x[k].shape[1]):
                for j in range(x[k].shape[0]):
                    f.write(str(x[k][j][i]))
                    f.write("\n")
                f.write(str(b[k][i]))
                f.write("\n")
    f.close()



