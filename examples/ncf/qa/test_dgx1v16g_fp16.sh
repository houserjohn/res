# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash
set -e
set -x

HARDWARE="DGX_1V_16G"
PRECISION="fp16"

./qa/utils/prepare_qa_dataset.sh

python -m torch.distributed.launch --nproc_per_node=8 ncf.py --data /ncf_data/cache/ml-20m --checkpoint_dir /ncf_data/checkpoints | tee nv.log
python -m logger.analyzer nv.log > nvlog.json

python qa/utils/compare.py --input nvlog.json --baseline ./qa/baseline_results/${HARDWARE}_${PRECISION}.json \
                           --keys best_train_throughput,best_eval_throughput,best_accuracy