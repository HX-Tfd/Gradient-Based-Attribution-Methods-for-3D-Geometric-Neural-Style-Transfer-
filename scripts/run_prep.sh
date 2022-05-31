#!/usr/bin/env bash

SCRIPT_PATH="../preprocess.py"

args=(
  --subdiv_iters 3
  --exp_name "lamp_parts"
  --working_dir "/Users/xihan/Desktop/BA/experiments/nst3d"
)

python $SCRIPT_PATH "${args[@]}"

