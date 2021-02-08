load("@rules_python_external//:defs.bzl", "pip_install")

def rtamt_pip_deps():
    excludes = native.existing_rules().keys()
    if "rtamt_pip_deps" not in excludes:
        pip_install(
            name = "rtamt_pip_deps",
            requirements = "@rtamt_deps//:requirements.txt",
        )
