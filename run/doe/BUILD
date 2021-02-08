load(":bundle.bzl", "doe_config_param", "doe_config")
load("@brezel//rules/doe:gke.bzl", "doe_gke")


doe_config_param(
    name = "algo",
    values = ["ddpg", "sac", "ppo", "td3"],
    always_contribute_to_naming = True,
)

doe_config_param(
    name = "layer",
    values = [
        "[{}]".format(", ".join([neuron for _ in range(layers)]))
        for layers in [1, 2, 3, 4]
        for neuron in ["4", "8", "16", "32", "64"]
    ],
)

doe_config_param(
    name = "n_steps",
    values = ["3000000"],
    naming = "N{}",
)

doe_config_param(
    name = "used_states",
    values = [
        "['thrust','p','q','r','e_p','e_q','e_r']",
        "['p','q','r','e_p','e_q','e_r']",
        "['e_p','e_q','e_r']",
    ],
    naming = "OBSERVED_{}",
)

doe_config_param(
    name = "activation_end_of_actor",
    values = ["tanh"],
)

doe_config(
    name = "config",
    params = [":algo", ":layer", ":used_states", ":activation_end_of_actor", ":n_steps"],
    substitutions = {
        "{GCS_BUCKET}": "gs://atcp-data/experiments/quadcopter", # this key is injected by bundle.bzl
        "{AERO}": "False",
        "{MAGNITUDE_MAX}": "1.",
        "{SATURATION}": "1.",
        "{PID_RATES}": "None",
        "{PID_THRUST}": "pid_thrust_main",
        # test.yaml.tpl:
        "{QUERY_CLASS}": "easy",
        "{EXP_DIR}": "training",
        "{PROPERTIES_OBSERVER_ENABLED}": "True",
        # observer.yaml.tpl:
        "{AGGREGATE_EPISODES}": "True",
        "{AGGREGATE_CHECKPOINTS}": "True",
        "{AGGREGATE_EXPERIMENTS}": "False",
        "{AGGREGATE_CHECKPOINT_ID}": "-1",
        "{TESTING_BASE_DIR}": "testing/",
   },
    visibility = ["//run:__pkg__"],
)

filegroup(
    name = "all-config-files",
    srcs = [
        ":config.files",
        "//run/doe/extra:config-saturation.files",
        "//run/doe/extra:config-windgust.files",
        "//run/doe/extra:config-saturation-nominal.files",
        "//run/doe/extra:config-windgust-nominal.files",
    ],
    visibility = ["//run:__pkg__"],
)

doe_gke(
    name = "doe-training",
    experiment = "quadcopter-training",
    image = {"eu.gcr.io/atcp-testing/quadcopter/training": "//run:training-img"},
    gcs_upload = {"/results/": "gs://atcp-data/experiments/quadcopter/training"},
    config = ":config",
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
)

doe_gke(
    name = "doe-testing",
    experiment = "quadcopter-testing",
    image = {"eu.gcr.io/atcp-testing/quadcopter/testing": "//run:test-img"},
    gcs_upload = {"/results/testing/": "gs://atcp-data/experiments/quadcopter/testing"},
    config = ":config",
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
)

doe_gke(
    name = "doe-observer",
    experiment = "quadcopter-observer",
    image = {"eu.gcr.io/atcp-testing/quadcopter/observer": "//run:observer-img"},
    gcs_upload = {"/results/observer/": "gs://atcp-data/experiments/quadcopter/observer"},
    config = ":config",
    nodepool = "pool-experiments",
    requests = {"cpu": "0.25"},
)
