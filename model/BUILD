load("@rules_python//python:defs.bzl", "py_library")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")

py_library(
    name = "model",
    srcs = glob(["*.py"], exclude=["*_test.py"]),
    deps = [
        "//utils",
        requirement("numpy"),
        project_requirement("matplotlib"),
        project_requirement("scipy"),
    ],
    visibility = ["//visibility:public"],
)

[py_test(
    name = test_file[:-3],
    srcs = [test_file],
    deps = [":model"],
) for test_file in glob(["*_test.py"])]
