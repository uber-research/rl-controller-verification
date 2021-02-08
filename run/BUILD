load("@rules_python//python:defs.bzl", "py_binary")
load("@python3_deps//:requirements.bzl", "requirement")
load("@io_bazel_rules_docker//python3:image.bzl", "py3_image")
load("@io_bazel_rules_docker//python:image.bzl", "py_layer")

package(default_visibility = ["//:__pkg__", "//run:__subpackages__"])


# TRAINING
# --------

TRAINING_DEPS = [
    "//architectures",
    "//modules",
    "//training",
    "//utils",
    "@brezel//:buildinfo",
]

TRAINING_DATA = [
    "@gym_envs//:gym-quadcopter",
    "//run/config:default",
]

py_binary(
    name = "training",
    main = "run_training.py",
    srcs = ["run_training.py"],
    deps = TRAINING_DEPS + ["@brezel//brezel/infra/doe:local"],
    data = TRAINING_DATA,
)

py3_image(
    name = "training-img",
    main = "run_training.py",
    srcs = ["run_training.py"],
    layers = [":training_layers"],
    imports = ["../../gym_envs/gym-quadcopter"], # Required for the gym env to be in the PYTHONPATH
    data = TRAINING_DATA + ["//run/doe:all-config-files"],
    base = "//docker:python3_openmpi_sm",
)
py_layer(
    name = "training_layers",
    deps = TRAINING_DEPS + ["@brezel//brezel/infra/doe:remote"]
)

# PLOTS
# -----

py_binary(
    name = "plots",
    main = "run_plots.py",
    srcs = ["run_plots.py"],
    deps = [
        "//training",
        "//utils",
    ],
)

# EXPORT
# ------

py_binary(
    name = "export_sherlock",
    main = "run_export_sherlock.py",
    srcs = ["run_export_sherlock.py"],
    deps = [
        "//architectures",
        "//utils",
    ],
)

# TEST
# ----

TEST_DEPS = [
    "//training",
    "//utils",
    "//modules",
    "@brezel//brezel/infra/gcp",
]

py_binary(
    name = "test",
    main = "run_test.py",
    srcs = ["run_test.py"],
    args = ["$(locations //run/config:default)"],
    deps = TEST_DEPS + ["@brezel//brezel/infra/doe:local"],
    data = TRAINING_DATA,
)

py3_image(
    name = "test-img",
    main = "run_test.py",
    srcs = ["run_test.py"],
    layers = [":test_layers"],
    imports = ["../../gym_envs/gym-quadcopter"],
    data = TRAINING_DATA + ["//run/doe:all-config-files"],
    base = "//docker:python3_openmpi_sm",
)
py_layer(
    name = "test_layers",
    deps = TEST_DEPS + ["@brezel//brezel/infra/doe:remote"]
)


# PROPERTIES OBSERVER
# -------------------

OBSERVER_DEPS = [
    requirement("dataclasses"),
    requirement("pyyaml"),
    "//utils",
    "//properties_observer",
    "@brezel//brezel/infra/gcp",
]

py_binary(
    name = "observer",
    main = "run_observer.py",
    srcs = ["run_observer.py"],
    deps = OBSERVER_DEPS + ["@brezel//brezel/infra/doe:local"],
    data = ["//run/config:default"],
)

py3_image(
    name = "observer-img",
    main = "run_observer.py",
    srcs = ["run_observer.py"],
    layers = [":observer_layers"],
    data = ["//run/config:default", "//run/doe:all-config-files"],
    base = "@brezel//docker:python3_base",
)
py_layer(
    name = "observer_layers",
    deps = OBSERVER_DEPS + ["@brezel//brezel/infra/doe:remote"]
)
