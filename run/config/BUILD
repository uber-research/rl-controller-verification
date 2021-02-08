load("//run/config:experiment.bzl", "experiment_config")

package(default_visibility = ["//run:__pkg__", "//training:__pkg__"])

exports_files(glob(["*.tpl"]))

experiment_config(
    name = "default",
    training = {
        "used_states": "['e_p','e_q','e_r']",
        "algo": "ddpg",
        "layer": "[16, 16]",
        "n_steps": "3000000",
        "aero" : "False",
        "magnitude_max": "1.",
        "saturation": "1.",
        "activation_end_of_actor": "tanh",
        "pid_rates": "None",
        "pid_thrust": "pid_thrust_main"
    },
    export = {
        "exp_dir": "experiment_20200608_145838",
    },
    test = {
        "query_class": "easy",
        "pid_rates": "None",
        "pid_thrust": "pid_thrust_main",
        "aero" : "False",
        "magnitude_max": "1.",
        "saturation": "1.",
        "exp_dir": "experiment_20200804_132745",
        "properties_observer_enabled" : "True",
    },
    observer = {
        "aggregate_episodes": "True",
        "aggregate_checkpoints": "True",
        "aggregate_experiments": "True",
        "aggregate_checkpoint_id": "-1",
        "testing_base_dir": "testing/",
    },
)
