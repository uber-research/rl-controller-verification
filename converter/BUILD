load("@rules_python//python:defs.bzl", "py_binary")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")

py_binary(
    name = "tflogs2pandas",
    srcs = ["tflogs2pandas.py"],
    deps = [
        extra_requirement("pandas"),
        extra_requirement("tensorboard"),
        project_requirement("tensorflow"),
    ],
)
