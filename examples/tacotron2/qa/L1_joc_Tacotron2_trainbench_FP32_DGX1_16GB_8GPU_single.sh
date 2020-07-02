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
       --epochs 6 \
       -bs 64 \
       --weight-decay 1e-6 \
       --grad-clip-thresh 1.0 \
       --cudnn-enabled \
       --anneal-steps 500 1000 1500 \
       --anneal-factor 0.3 \
       --load-mel-from-disk \
       --training-files filelists/ljs_mel_text_train_subset_2500_filelist.txt \
       --validation-files filelists/ljs_mel_text_val_filelist.txt \
       --dataset-path ./ \
       --log-file $LOG_FILE

if [ "$CHECK_CURVES" = "1" ]
then
    python qa/check_curves.py \
	   -b qa/tacotron2_fp32_8gpu-perf.json \
	   -g "epoch" \
	   -k "train_epoch_avg_items/sec" \
	   --skip 5 \
	   --damping 1 \
	   --eps 0 \
	   $LOG_FILE \
	   --fail low
fi
