#!/bin/bash

date

# Run first 2 in parallel (one per GPU)
CUDA_VISIBLE_DEVICES=1 python evaluate.py &
wait
CUDA_VISIBLE_DEVICES=1 python evaluate_copy.py &
wait
CUDA_VISIBLE_DEVICES=1 python evaluate_copy2.py &
wait

date
echo "All 3 experiments done"

# run with: nohup ./run_evals.sh > eval.log 2>&1 &