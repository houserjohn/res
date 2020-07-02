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
       --epochs 6 \
       -bs 10 \
       --segment-length 8000 \
       --weight-decay 0 \
       --grad-clip-thresh 65504.0 \
       --cudnn-enabled \
       --cudnn-benchmark \
       --training-files filelists/ljs_audio_text_train_subset_625_filelist.txt \
       --dataset-path ./ \
       --amp-run \
       --log-file $LOG_FILE    

if [ "$CHECK_CURVES" = "1" ]
then
    python qa/check_curves.py \
	   -b qa/waveglow_amp_8gpu-perf.json \
	   -g "epoch" \
	   -k "train_epoch_avg_items/sec" \
	   --skip 5 \
	   --damping 1 \
	   --eps 0 \
	   $LOG_FILE \
	   --fail low
fi
