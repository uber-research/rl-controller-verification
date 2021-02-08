FROM brzl/quadcopter:base
USER root

# Install extra dependencies
# - OpenMPI
# - LibSM
RUN apt-get update --quiet && \
    apt-get -y install --quiet --assume-yes \
    libsm6 libxext6 libxrender-dev \
    ssh \
    && rm -rf /var/lib/apt/lists/*

# Installing OpenMPI from source to make sure we get version 4
COPY third_party/brezel/scripts/install/install_openmpi.sh /tmp
RUN /tmp/install_openmpi.sh

# Install pip3 (required by pip_local_install)
RUN apt-get update --quiet && \
    apt-get -y install --quiet --assume-yes \
    python3-dev python-pip python3-pip python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Install rtamt dependencies
# - CMake 3.17
# - Boost
# - Ninja
# - antlr4
ARG cmake_version=3.17.4
ARG cmake_sha256=126cc8356907913787d4ff35237ae1854c09b927a35dbe5270dd571ae224bdd3
COPY third_party/brezel/scripts/install/install_cmake.sh /tmp
RUN /tmp/install_cmake.sh ${cmake_version} ${cmake_sha256}
RUN apt-get update -q && apt-get install -y \
    antlr4 \
    libboost-all-dev \
    ninja-build

# Install X11 utils
RUN apt-get update -q && apt-get install -q -y \
    xterm \
    x11-xserver-utils

# Prepare results folders
RUN mkdir -p /checkpoints /results && \
    chown -R ubrzl:ubrzl /results /checkpoints

# Update working directory
WORKDIR /quadcopter
USER ubrzl
