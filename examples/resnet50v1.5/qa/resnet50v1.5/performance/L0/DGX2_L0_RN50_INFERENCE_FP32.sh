python ./qa/testscript.py /imagenet --raport `basename ${0} .sh`_raport.json --workspace $1 $2 -j 3 --data-backends syntetic --bench-iterations 100 --bench-warmup 3 --epochs 1 --arch resnet50 -c fanin --label-smoothing 0.1 --mixup 0.0 --mode inference --ngpus 1  --bs 128