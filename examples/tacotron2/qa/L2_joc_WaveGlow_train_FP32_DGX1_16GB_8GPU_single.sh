#!/usr/bin/env bash

set -e

CHECK_CURVES="1"
LOG_FILE="nvlog.json"

if [ "$#" -eq 2 ]
then
    CHECK_CURVES=$1
    LOG_FILE=$2
fi


python -m multiproc train.py \
       -m WaveGlow \
       -o ./ \
       -lr 1e-4 \
       --epochs 10 \
       -bs 4 \
       --segment-length 8000 \
       --weight-decay 0 \
       --grad-clip-thresh 65504.0 \
       --cudnn-enabled \
       --cudnn-benchmark \
       --dataset-path ./ \
       --log-file $LOG_FILE

if [ "$CHECK_CURVES" = "1" ]
then
    python qa/check_curves.py \
	   -b qa/waveglow_fp32-short-loss.json \
	   -g epoch \
	   --skip 9 \
	   --damping 1 \
	   --perc -0.15 \
	   -k "train_epoch_avg_loss" \
	   $LOG_FILE \
	   --fail high
fi
