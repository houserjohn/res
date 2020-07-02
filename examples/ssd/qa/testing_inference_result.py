# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import sys
import numpy as np

import examples.SSD300_inference as inf

expected_result = np.array([
     [np.array([[0.34423554, 0.62172854, 0.435952  , 0.6657095 ],
                [0.09019691, 0.6279105 , 0.15782164, 0.7066598 ],
                [0.63514817, 0.13939235, 0.91875887, 0.7972523 ]], dtype=np.float32),
      np.array([46, 42,  1]),
      np.array([0.32345006, 0.40823278, 0.9874778 ], dtype=np.float32)],
     [np.array([[0.3351106, 0.534543 , 0.5769685, 0.8397729]], dtype=np.float32),
      np.array([70]),
      np.array([0.76288354], dtype=np.float32)],
     [np.array([[0.22057474, 0.5427369 , 0.5059987 , 0.8617027 ],
                [0.53903955, 0.10348529, 0.6878695 , 0.24030215],
                [0.5081272 , 0.39717296, 0.6809095 , 0.87010217]], dtype=np.float32),
      np.array([57, 10,  1]),
      np.array([0.30137864, 0.78862876, 0.9993974 ], dtype=np.float32)]])


def close(result, expected):
    return result.shape == expected.shape and np.allclose(np.concatenate(result), np.concatenate(expected))


def main():
    res = inf.main([
        'http://images.cocodataset.org/val2017/000000397133.jpg',
        'http://images.cocodataset.org/val2017/000000037777.jpg',
        'http://images.cocodataset.org/val2017/000000252219.jpg',
        ])
    res = np.array(res)

    bboxes      = res[:,0]
    classes     = res[:,1]
    confidences = res[:,2]

    if not close(np.concatenate(bboxes), np.concatenate(expected_result[:,0])) or \
       not close(classes, expected_result[:,1]) or \
       not close(confidences, expected_result[:,2]):
           print ('FAILED')
           sys.exit(-1)

    print('PASSED')


if __name__ == '__main__':
    main()
