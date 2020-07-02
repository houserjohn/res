#!/bin/bash

python ./qa/benchmark_performance.py --benchmark-mode training --ngpus 1 4 8 --bs 32 --bench-warmup 100 --bench-iterations 200 --benchmark-file qa/benchmark_baselines/SSD300_pytorch_19.08_training_fp32.json --data $1
