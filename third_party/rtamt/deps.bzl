load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

_build = """
load("@rtamt_pip_deps//:requirements.bzl", "requirement")
filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"]
)
filegroup(
    name = "rtamt",
    srcs = glob(["rtamt/**"]),
    visibility = ["//visibility:public"]
)
alias(
    name = "antlr4",
    actual = requirement("antlr4-python3-runtime"),
    visibility = ["//visibility:public"]
)
"""


def rtamt_deps():
    """Create bazel repository for rtamt

    Two patches are immediately applied in the cloned rtamt repository:
    - CMakeLists.txt is edited to fix an issue during cmake build.
    - Version of antlr4 is extracted from setup.py to create requirements.txt
      which is read by '@rtamt_deps//:antlr4'.
    """
    new_git_repository(
        name = "rtamt_deps",
        remote = "https://github.com/nickovic/rtamt",
        commit = "5366349c44afb53cffa5fb29e43fea1eb23b6c52",
        shallow_since = "1592029532 +0200",
        patch_cmds = [
            "sed -i 's/-py35//' rtamt/cpplib/stl/rtamt_stl_library_wrapper/CMakeLists.txt",
            "grep -Po 'antlr4-python3-runtime==([0-9\.]*)' setup.py > requirements.txt",
        ],
        build_file_content = _build,
    )
