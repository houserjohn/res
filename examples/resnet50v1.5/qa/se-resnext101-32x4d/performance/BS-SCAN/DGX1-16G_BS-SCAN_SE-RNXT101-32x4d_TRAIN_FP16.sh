python ./qa/testscript.py /imagenet --raport `basename ${0} .sh`_raport.json --workspace $1 $2 -j 5 --data-backends syntetic --bench-iterations 10 --bench-warmup 1 --epochs 1 --bs 1 2 4 8 16 32 48 64 96 112 128 160 192 224 256 --ngpus 1 --arch se-resnext101-32x4d -c fanin --label-smoothing 0.1 --mixup 0.0 --mode training --fp16 --static-loss-scale 128