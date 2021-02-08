load("@brezel//rules:images.bzl", "extended_python_image")

## Create docker image with python3, openmpi and libSM
extended_python_image(
    name = "python3_openmpi_sm",
    packages = [
        "libopenmpi-dev",
        "libsm6",
        "libgl1-mesa-glx",  # Required by stable-baselines's opencv dependency
        "libglib2.0-0",     # Required by stable-baselines's opencv dependency
        "libxext6",
        "libxrender-dev",
        "ssh"
    ],
    env = {
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/openmpi/lib",
    },
    visibility = ["//visibility:public"],
)
