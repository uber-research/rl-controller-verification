load("//run/doe:bundle.bzl", "doe_config", "doe_config_param")
load("@brezel//rules/doe:gke.bzl", "doe_gke")

MODES = ["saturation", "windgust"]
CFGS = {
    "nominal": {
        "{AERO}": "False",
        "{MAGNITUDE_MAX}": "1.",
        "{SATURATION}": "1.0",
        "{USED_STATES}": "['e_p','e_q','e_r']",
    },
    "saturation": {
        "{AERO}": "False",
        "{MAGNITUDE_MAX}": "1.",
        "{SATURATION}": "0.8",
        "{USED_STATES}": "['e_p','e_q','e_r','saturation']",
    },
    "windgust": {
        "{AERO}": "True",
        "{MAGNITUDE_MAX}": "10.",
        "{SATURATION}": "1.",
        "{USED_STATES}": "['e_p','e_q','e_r','wg_x','wg_y','wg_z','wg_a']",
    }
}

doe_config_param(
    name = "algo",
    values = ["ddpg", "sac"],
)

doe_config_param(
    name = "layer",
    values = ["[16, 16]", "[32, 32]"],
)

doe_config_param(
    name = "n_steps",
    values = ["3000000"],
    naming = "N{}",
)

doe_config_param(
    name = "activation_end_of_actor",
    values = ["tanh"],
)

[doe_config_param(
    name = "us-{}".format(cfg),
    key = "USED_STATES",
    values = [CFGS[cfg]["{USED_STATES}"]],
    naming = "OBSERVED_{}",
) for cfg in MODES]

[doe_config(
    name = "config-{}{}".format(cfg, "-nominal" if nominal else ""),
    params = [":algo", ":layer", ":us-{}".format(cfg), ":activation_end_of_actor", ":n_steps"],
    substitutions = {
        "{AERO}": CFGS["nominal" if nominal else cfg]["{AERO}"],
        "{MAGNITUDE_MAX}": CFGS["nominal" if nominal else cfg]["{MAGNITUDE_MAX}"],
        "{SATURATION}": CFGS["nominal" if nominal else cfg]["{SATURATION}"],
        "{GCS_BUCKET}": "gs://atcp-data/experiments/quadcopter", # this key is injected by bundle.bzl
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
    visibility = ["//run:__pkg__", "//run/doe:__pkg__"],
) for cfg in MODES for nominal in [True, False]]

# TRAINING
[doe_gke(
    name = "doe-training-{}".format(cfg),
    experiment = "quadcopter-training",
    image = {"eu.gcr.io/atcp-testing/quadcopter/training": "//run:training-img"},
    gcs_upload = {"/results/": "gs://atcp-data/experiments/quadcopter/training"},
    config = ":config-{}".format(cfg),
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
) for cfg in MODES]

# TESTING
[doe_gke(
    name = "doe-testing-{}".format(cfg),
    experiment = "quadcopter-testing",
    image = {"eu.gcr.io/atcp-testing/quadcopter/testing": "//run:test-img"},
    gcs_upload = {"/results/testing/": "gs://atcp-data/experiments/quadcopter/testing"},
    config = ":config-{}".format(cfg),
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
) for cfg in MODES]

[doe_gke(
    name = "doe-testing-{}-nominal".format(cfg),
    experiment = "quadcopter-testing-nominal",
    image = {"eu.gcr.io/atcp-testing/quadcopter/testing": "//run:test-img"},
    gcs_upload = {"/results/testing/": "gs://atcp-data/experiments/quadcopter/testing/NOMINAL"},
    config = ":config-{}-nominal".format(cfg),
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
) for cfg in MODES]

# OBSERVER
[doe_gke(
    name = "doe-observer-{}".format(cfg),
    experiment = "quadcopter-observer",
    image = {"eu.gcr.io/atcp-testing/quadcopter/observer": "//run:observer-img"},
    gcs_upload = {"/results/observer/": "gs://atcp-data/experiments/quadcopter/observer"},
    config = ":config-{}".format(cfg),
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
) for cfg in MODES]

[doe_gke(
    name = "doe-observer-{}-nominal".format(cfg),
    experiment = "quadcopter-observer-nominal",
    image = {"eu.gcr.io/atcp-testing/quadcopter/observer": "//run:observer-img"},
    gcs_upload = {"/results/observer/": "gs://atcp-data/experiments/quadcopter/observer/NOMINAL"},
    config = ":config-{}-nominal".format(cfg),
    nodepool = "pool-experiments",
    requests = {"cpu": "0.75"},
) for cfg in MODES]
