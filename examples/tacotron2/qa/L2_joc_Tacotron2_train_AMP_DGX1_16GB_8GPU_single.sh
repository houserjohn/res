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
       -m Tacotron2 \
       -o ./ \
       -lr 1e-3 \
       --epochs 10 \
       -bs 128 \
       --weight-decay 1e-6 \
       --grad-clip-thresh 1.0 \
       --cudnn-enabled \
       --anneal-steps 500 1000 1500 \
       --anneal-factor 0.1 \
       --amp-run \
       --dataset-path ./ \
       --log-file $LOG_FILE

if [ "$CHECK_CURVES" = "1" ]
then
    python qa/check_curves.py \
	   -b qa/tacotron2_amp-short-loss.json \
	   -g epoch \
	   --skip 9 \
	   --damping 1 \
	   -k "train_epoch_avg_loss" \
	   $LOG_FILE \
	   --fail high
fi
