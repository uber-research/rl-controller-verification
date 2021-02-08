#!/bin/bash
make build
export CHECKPOINTS_VOLUME=checkpoints
export RESULTS_VOLUME=results
make test
