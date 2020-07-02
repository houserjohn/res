#!/bin/bash

set -e

CHECK_CURVES="1"
LOG_FILE="nvlog.json"

if [ "$#" -eq 2 ]
then
    CHECK_CURVES=$1
    LOG_FILE=$2
fi


## uncomment to generate new baseline; will be created in qa/baselines/ ##
## python inference_perf.py -m Tacotron2 -bs=20  --create-benchmark

python inference_perf.py \
       -m Tacotron2 \
       -bs=1 \
       --amp-run \
       --decoder-no-early-stopping \
       --log-file $LOG_FILE

if [ "$CHECK_CURVES" = "1" ]
then
    python qa/check_curves.py \
	   -b qa/tacotron2_amp-infer.json \
	   $LOG_FILE \
	   -g "iter" \
	   -k "items_per_sec" \
	   --eps 0.001 \
	   --damping 1 \
	   --perc 0.4 \
	   --fail low
fi
