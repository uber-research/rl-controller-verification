Learning a Control for Quadcopter with Reinforcement Learning
=============================================================


Development environment
-----------------------

> __WARNING__ The Dockerfile on this project uses a base image which has been hijacked.  
> The Dockerfile has been modified to comment out the hijacked base.  If you wish to use
> the base image, please do so with caution.

A docker image is provided for the code development: `brzl/quadcopter:devel`.
The definition of the image can be found inside the folder `docker`. As you
can see, it is based on the brezel research platform, with additional python
dependencies. The first thing you need to do is build the image, which can
takes a while (but you usually need to build it once for all):

    make build


The commands described in the rest of this README are supposed to be run
from the devel container. Enter `make run` to get a shell.

If you need X11 suport, use `make run-x11` instead.
Do `xrdb -merge Xresources` once in the container.


Getting started
---------------

#### 1. Build applications

    bazel build :all

Note that you will need to run `bazel sync --configure` in case you
modify the content of folder `gym-quadcopter`.

#### 2. Run the training

    bazel run :training

It will read params from `config_training.yaml` file.
Results will be available in `results/experiment_DATETIME`.

#### 3. Inspect results with Tensorboard

You need to enter the Tensorboard Container (from your host) with

    make tensorboard

and then run Tensorboard making it point to the experiment directory

    tensorboard --logdir /results/experiment_DATETIME/runs --bind_all

#### 4. Export Weights for Sherlock

    bazel run :export

It will read params from `config_export.yaml` file.

The input is the `model/quadcopter-{i}.pkl` selected
and the export is defined in the config file.

Notes
-----

- `config_training.yaml` has all hyperparameters
- in `models`: the pkl checkpoints
- in `figures`: all plots (rewards etc.)
- in `log`: log file
- in `tests`: all tests that are run with proper tags
- `config_tests` contains the corresponding parameters and figures and log inside each of them
