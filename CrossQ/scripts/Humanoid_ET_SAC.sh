#!/bin/bash

seeds=(1 2 3 4)

for seed in "${seeds[@]}"
do
  echo "Running with seed $seed"
  python train.py \
    -algo sac \
    -env Humanoid-v4 \
    -seed $seed \
    -wandb_mode 'disabled' \
    -actor_use_bn True \
    -update_mode ETETT \
    -bn_mode bn
done