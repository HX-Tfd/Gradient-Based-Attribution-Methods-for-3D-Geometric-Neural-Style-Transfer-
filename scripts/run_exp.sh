#!/usr/bin/env bash

SCRIPT_PATH="experiment_scripts/train_sdf.py"


args=(
  --model_type "sine"
  --point_cloud_path "preprocessed_meshes/bag2/bag2_0.xyzn"
  --batch_size 1000
  --experiment_name "bag_experiment"
  --steps_til_summary 1000
  --epochs_til_ckpt 1000
  --num_epochs 50000
)

python $SCRIPT_PATH "${args[@]}"
