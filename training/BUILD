load("@rules_python//python:defs.bzl", "py_library")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")

py_library(
    name = "training",
    srcs = glob(["*.py"], exclude=["*_test.py"]),
    deps = [
        "//architectures",
        "//model",
        "//utils",
        requirement("dataclasses"),
        requirement("numpy"),
        project_requirement("cloudpickle"),
        project_requirement("deprecation"),
        project_requirement("gym"),
        project_requirement("matplotlib"),
        project_requirement("scipy"),
        project_requirement("tensorflow"),

    ],
    visibility = ["//visibility:public"],
)

[py_test(
    name = test_file[:-3],
    srcs = [test_file],
    deps = [":training", "//modules", "@brezel//:buildinfo"],
    data = ["//run/config:default"],
) for test_file in glob(["*_test.py"])]
