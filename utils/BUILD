load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")

py_library(
    name = "utils",
    srcs = glob(["*.py"], exclude=["*_test.py"]),
    deps = [
        requirement("dash"),
        requirement("numpy"),
        requirement("pyyaml"),
        requirement("dataclasses"),
        requirement("google-cloud-storage"),
        project_requirement("cloudpickle"),
        project_requirement("gym"),
        project_requirement("matplotlib"),
        project_requirement("scipy"),
        project_requirement("stable_baselines"),
        project_requirement("tensorflow"),
    ],
    visibility = ["//visibility:public"],
)

[py_test(
    name = test_file[:-3],
    srcs = [test_file],
    deps = [":utils", "//modules"],
) for test_file in glob(["*_test.py"])]
