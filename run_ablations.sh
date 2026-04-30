#!/bin/bash
# Run first 2 in parallel (one per GPU)
CUDA_VISIBLE_DEVICES=0 python train.py &
wait
CUDA_VISIBLE_DEVICES=1 python train_copy.py &

# Wait for both to finish
wait

# Run next 2 in parallel
CUDA_VISIBLE_DEVICES=0 python train_copy_2.py &
wait
CUDA_VISIBLE_DEVICES=1 python train_copy_3.py &

# Wait for both to finish
wait

echo "All 4 experiments done"

# run with: nohup ./run_ablations.sh > ablation_log.txt 2>&1 &