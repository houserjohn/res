# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


import json
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--baseline', type=str, required=True)
parser.add_argument('--keys', type=str, required=True, help='Comma-separated list of keys to compare')
parser.add_argument('--tolerance', type=float, default=0.9)

args = parser.parse_args()

args.keys = args.keys.split(',')

input = json.load(open(args.input))
baseline = json.load(open(args.baseline))

fail = False
for key in args.keys:
    print('Checking key:', key)
    input_value = input['run'][key]
    baseline_value = baseline['run'][key]
    print("input value: ", input_value)
    print("baseline value: ", baseline_value)

    # only 'higher is better'-type scenarios currently supported
    if input_value < 0.9 * baseline_value:
        print('FAILED\n')
        fail = True
    else:
        print('PASSED\n')

if fail:
    sys.exit(1)
else:
    sys.exit(0)
