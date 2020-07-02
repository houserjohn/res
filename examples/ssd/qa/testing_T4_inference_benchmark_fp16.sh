#!/bin/bash

python ./qa/benchmark_performance.py --benchmark-mode inference --ngpus 1 --bs 1 2 4 8 --amp  --bench-warmup 100 --bench-iterations 200 --benchmark-file qa/benchmark_baselines/SSD300_pytorch_19.08_T4_inference_fp16.json --data $1
