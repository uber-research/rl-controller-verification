#!/usr/bin/env bash
for i in 0 1 2 4 5
do
    xterm -T "env${i}" -e bazel run '//:training' "config${i}.yaml" &
done
