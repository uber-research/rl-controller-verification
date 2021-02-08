load("@python3_deps//:requirements.bzl", "requirement")
load("@python3_extra_deps//:requirements.bzl", extra_requirement="requirement")
load("@python3_project_deps//:requirements.bzl", project_requirement="requirement")

py_library(
    name = "python_scripts",
    srcs = glob(["*.py"]),
    deps = [
        requirement("numpy"),
        extra_requirement("pandas"),
        project_requirement("hiplot"),
        project_requirement("matplotlib"),
        project_requirement("numba"),
        project_requirement("py7zr"),
    ],
    visibility = ["//visibility:public"],
)
