#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
#
# HF-hub vs local-natten parity for the hopper-fna backend (na1d/na2d/na3d).
# Skipped unless running on a Hopper (SM90) GPU.

import unittest

import torch
from natten.utils.testing import (
    skip_if_hopper_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
)

from .utils import random_fna_problem, reset_everything, run_fna_case

BACKEND = "hopper-fna"
DTYPES = [torch.float16, torch.bfloat16]

# (batch, heads, heads_kv, head_dim, input_shape, kernel_size, stride, dilation)
PROBLEMS_1D = [
    (2, 4, 4, 128, (128,), (63,), (31,), (1,)),
    (1, 1, 1, 128, (2048,), (2048,), (1,), (1,)),
]
PROBLEMS_2D = [
    (1, 1, 1, 128, (19, 29), (8, 8), (1, 1), (2, 3)),
    (1, 1, 1, 64, (28, 40), (17, 31), (1, 1), (1, 1)),
]
PROBLEMS_3D = [
    (1, 4, 1, 32, (8, 8, 16), (3, 3, 3), (2, 1, 2), (2, 2, 4)),
]


class HopperFNAParityTest(unittest.TestCase):
    def _run_problems(self, problems, na_dim):
        compared = 0
        for i, (b, h, hkv, d, shape, ks, st, di) in enumerate(problems):
            for is_causal in (tuple([False] * na_dim), tuple([True] * na_dim)):
                reset_everything(seed=i, deterministic=False)
                compared += run_fna_case(
                    backend=BACKEND,
                    batch=b,
                    heads=h,
                    heads_kv=hkv,
                    head_dim=d,
                    input_shape=shape,
                    kernel_size=ks,
                    stride=st,
                    dilation=di,
                    is_causal=is_causal,
                    dtypes=DTYPES,
                    deterministic=False,
                    configs_to_test=3,
                )
        if compared == 0:
            self.skipTest(
                "no launchable configs for this backend on this GPU/hub build"
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_1d(self):
        self._run_problems(PROBLEMS_1D, 1)

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_2d(self):
        self._run_problems(PROBLEMS_2D, 2)

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_3d(self):
        self._run_problems(PROBLEMS_3D, 3)

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep(self):
        compared = 0
        for na_dim in (1, 2, 3):
            for i in range(6):
                reset_everything(seed=3000 * na_dim + i, deterministic=False)
                p = random_fna_problem(na_dim, BACKEND)
                compared += run_fna_case(
                    backend=BACKEND, dtypes=DTYPES, deterministic=False, **p
                )
        if compared == 0:
            self.skipTest(
                "no launchable configs for this backend on this GPU/hub build"
            )


if __name__ == "__main__":
    unittest.main()
