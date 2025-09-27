#!/bin/bash

seeds=(1 2 3 4)

for seed in "${seeds[@]}"
do
  echo "Running with seed $seed"
  python train.py \
    -algo crossq \
    -env Walker2d-v4 \
    -seed $seed \
    -wandb_mode 'disabled' \
    -update_mode ETETT \
    -eval_qbias 1
done