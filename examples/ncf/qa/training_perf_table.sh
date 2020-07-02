#! /bin/bash
set -e
set -x

./qa/utils/prepare_qa_dataset.sh

for gpus in 1 8 16; do
    for precision in FP16 FP32; do
        for seed in $(seq 1 5); do
            echo "num gpus: ${gpus}, precision: ${precision}"
            if [ ${precision}  == "FP16" ]; then
                python -m torch.distributed.launch --nproc_per_node=${gpus} ncf.py \
                    --data /ncf_data/cache/ml-20m --seed ${seed} | tee nv.log
            else
                python -m torch.distributed.launch --nproc_per_node=${gpus} ncf.py \
                    --data /ncf_data/cache/ml-20m --opt_level O0 --seed ${seed} | tee nv.log
            fi

            python -m logger.analyzer nv.log > training_${gpus}_${precision}_${seed}.json
        done
    done
done

pip install tabulate
python qa/utils/train_perf_table_analysis.py