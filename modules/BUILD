load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "modules",
    srcs = glob(["**/*.py"], exclude=["*_test.py"]),
    visibility = ["//visibility:public"],
)

[py_test(
    name = test_file[:-3],
    srcs = [test_file],
    deps = [":modules"],
) for test_file in glob(["**/*_test.py"])]
