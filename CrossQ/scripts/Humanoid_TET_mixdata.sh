#!/bin/bash

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"
do
  echo "Running with seed $seed"
  python train.py \
    -algo crossq \
    -env Humanoid-v4 \
    -seed $seed \
    -wandb_mode 'disabled' \
    -update_mode TETTT \
    -mix_data True \
    -eval_qbias 1
done