load("@rules_python//python:defs.bzl", "py_library")
load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")

py_library(
    name = "properties_observer",
    srcs = glob(["*.py"], exclude=["*_test.py"]),
    deps = [
        requirement("numpy"),
        project_requirement("pandas"),
        project_requirement("numba"),
        project_requirement("matplotlib"),
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "test_data",
    testonly = 1,
    srcs = glob(["data/*"]),
    visibility = ["//visibility:public"],
)

[py_test(
    name = test_file[:-3],
    srcs = [test_file],
    deps = [":properties_observer",],
    data = [":test_data"],
) for test_file in glob(["*_test.py"])]
