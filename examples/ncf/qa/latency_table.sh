#! /bin/bash
set -e
set -x

for batch_size in 26744 53488 106976 213952 427904 ; do
    for precision in FP16 FP32; do
        echo "batch: ${batch_size}, precision: ${precision}"
        if [ ${precision}  == "FP16" ]; then
            python inference.py --num_batches 1000 --batch_size ${batch_size} | tee nv.log
        else
            python inference.py --num_batches 1000 --batch_size ${batch_size} --opt_level O0 | tee nv.log
        fi

        python -m logger.analyzer nv.log > inference_${batch_size}_${precision}.json
    done
done

python qa/utils/latency_data_analysis.py