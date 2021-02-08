from setuptools import setup

setup(
    name='gym_quadcopter',
    version='0.0.1',
    install_requires=['gym'],
    packages=["gym_quadcopter", "gym_quadcopter.envs"],
    include_package_data=True
)
