# Copyright 2026 FlagOS Contributors
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

"""Ad-hoc fair-comparison probe for trinary perf.

Measures three flavors of trinary execution on identical inputs in `operator`
mode host-loop style (matches benchmark_core.time_operator):
  A) flagtensor.cutensor.trinary wrapper (constructs a fresh executor per call)
  B) CuTensorTrinary executor reused across calls (the apples-to-apples form)
  C) flagtensor.trinary (our Triton path; kernel autotuned + cached)

Prints a table so we can see where the ~4ms baseline latency in the current
benchmark actually comes from.
"""
import time

import torch

from flagtensor import trinary as flag_trinary
from flagtensor.cutensor import CuTensorTrinary, trinary as cutensor_trinary


def host_loop(fn, *, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / reps * 1000.0


def probe(shape, dtype, *, warmup=50, reps=100):
    a = torch.empty(shape, device="cuda", dtype=dtype).uniform_(0.5, 4.0)
    b = torch.empty(shape, device="cuda", dtype=dtype).uniform_(-2.0, 2.0)
    c = torch.empty(shape, device="cuda", dtype=dtype).uniform_(0.5, 4.0)

    def call_wrapper():
        return cutensor_trinary(
            a, b, c,
            op_a="log", op_b="neg", op_c="sqrt",
            op_ab="add", op_abc="max",
        )

    executor = CuTensorTrinary(
        op_ab="add", op_abc="max",
        op_a="log", op_b="neg", op_c="sqrt",
        dtype=dtype,
    )

    def call_persistent():
        return executor(a, b, c)

    def call_triton():
        return flag_trinary(
            a, b, c,
            op_a="log", op_b="neg", op_c="sqrt",
            op_ab="add", op_abc="max",
        )

    lat_wrapper = host_loop(call_wrapper, warmup=warmup, reps=reps)
    lat_persistent = host_loop(call_persistent, warmup=warmup, reps=reps)
    lat_triton = host_loop(call_triton, warmup=warmup, reps=reps)

    return lat_wrapper, lat_persistent, lat_triton


def main():
    configs = [
        ((1024,), torch.float16),
        ((4096,), torch.float16),
        ((1024,), torch.float32),
        ((4096,), torch.float32),
        ((2, 3, 32, 64, 16), torch.float16),
        ((2, 3, 32, 64, 16), torch.float32),
    ]

    print(f"{'shape':<24} {'dtype':<12} {'cutensor wrapper':>18} {'cutensor persistent':>22} {'triton':>12} {'speedup(pers)':>14}")
    for shape, dtype in configs:
        lw, lp, lt = probe(shape, dtype)
        shape_str = str(shape)
        dtype_str = str(dtype).replace("torch.", "")
        speedup_pers = lp / lt if lt > 0 else float("inf")
        print(
            f"{shape_str:<24} {dtype_str:<12} "
            f"{lw:>16.4f}ms {lp:>20.4f}ms {lt:>10.4f}ms {speedup_pers:>13.2f}x"
        )


if __name__ == "__main__":
    main()
