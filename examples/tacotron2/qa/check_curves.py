# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import json
import numpy as np
import sys


def crop_length(*args):
    min_len = min([len(a) for a in args])
    return tuple([a[:min_len] for a in args])


def load_benchmark(path):
    with open(path) as file:
        data = json.loads(file.read())
    tile_size = data["tile_size"]
    data = np.array([np.array(d) for d in data["data"]])
    return data, tile_size


def check_threshold(data, mean, perc, skip, fail_high):
    data = data[skip:]
    mean = mean[skip:]
    thres = mean*(1.0+perc) if fail_high else mean*(1.0-perc)
    aggr = np.argmax if fail_high else np.argmin
    comp = (lambda v, t: v > t) if fail_high else (lambda v, t: v < t)
    ix = aggr(data)
    val = data[ix]

    if comp(val, thres):
        return False, (val, data[ix], mean[ix], thres)
    return True, (val, data[ix], mean[ix], thres[ix])


def smooth_exp(data, damping):
    mean_b = 0.
    result = []
    for v in data:
        mean_b_ = mean_b * (1. - damping) + v * damping
        result.append(mean_b_)
        mean_b = mean_b_
    result = np.array(result)
    return result


def tile_mean(d, tile_size):
    d = np.array(d)
    tile_n = len(d) // tile_size
    d = d[:tile_n * tile_size]
    d = d.reshape(tile_n, tile_size)
    d = d.mean(-1)
    return d


def main():
    parser = argparse.ArgumentParser(description="nvlog curve checker")
    parser.add_argument("--benchmark", "-b", required=True,
                        help="Path to benchmark data.")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--perc", type=float, default=0.1,
                        help="Fraction threshold for failure")
    parser.add_argument("--damping", type=float, default=0.001,
                        help="Damping factor for exponential average")
    parser.add_argument("--eps", type=float, default=0.001,
                        help="Minimal value for std to avoid division by zero")
    parser.add_argument("--key", "-k", type=str, required=True,
                        help="Key to extract from nvlog[\"iter\"]")
    parser.add_argument("--group", "-g", type=str, required=True,
                        help="Which nvlog group to check for key", choices=("iter", "epoch"))
    parser.add_argument("--skip", type=int, default=0,
                        help="Number of iterations to skip during comparison")
    parser.add_argument("--fail", "-f", type=str, required=True,
                        help="Whether to check for values which are too low or too high", choices=("low", "high"))
    args = parser.parse_args(sys.argv[1:])
    fail_high = args.fail == "high"

    benchmark, tile_size = load_benchmark(args.benchmark)
    benchmark = np.array([
        smooth_exp(v, args.damping)
        for v in benchmark
    ])
    mean, std = benchmark.mean(0), benchmark.std(0) + args.eps

    ok = True
    outlier = None
    for f in args.files:
        with open(f) as file:
            data = json.loads(file.read())
        data = data[args.group][args.key]
        if args.group == "iter":
            data = np.array([v for l in data for v in l])
        data = tile_mean(data, tile_size)
        data = smooth_exp(data, args.damping)
        data, mean_crop, std_crop = crop_length(data, mean, std)
        ok_, outlier_ = check_threshold(
            data, mean_crop, args.perc, args.skip, fail_high)
        ok = ok and ok_
        if outlier is None:
            outlier = outlier_
        else:
            if np.abs(outlier_[0]) > np.abs(outlier[0]):
                outlier = outlier_

    threshold = 1.0 + args.perc if fail_high else 1.0 - args.perc

    print(f"value: {outlier[1]}")
    print(f"mean: {outlier[2]}")
    print(f"threshold: {outlier[3]}")
    if ok:
        comp_sign = "<" if fail_high else ">"
        print(f'value = {outlier[1]} {comp_sign} threshold = {outlier[3]}, mean = {outlier[2]}, ')
        print("&&&& PASSED")
        sys.exit(0)
    else:
        comp_sign = ">" if fail_high else "<"
        print(f'value = {outlier[1]} {comp_sign} threshold = {outlier[3]}, mean = {outlier[2]}, ')
        print("&&&& FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
