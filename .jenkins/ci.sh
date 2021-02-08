#!/bin/bash
# Continuous integration test
# Called inside the container
bazel test --deleted_packages='//papers/hscc_2021/rep/instructions,//papers/hscc_2021/src' //...
