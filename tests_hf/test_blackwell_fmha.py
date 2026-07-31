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
# HF-hub vs local-natten parity for the blackwell-fmha backend (attention).

import unittest

import torch
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
)

from .utils import random_fmha_problem, reset_everything, run_fmha_case

BACKEND = "blackwell-fmha"
DTYPES = [torch.float16, torch.bfloat16]

# (batch, heads, heads_kv, head_dim, seqlen_q, seqlen_kv). Blackwell FMHA supports GQA/MQA but
# not head_dim_v != head_dim.
PROBLEMS = [
    (1, 1, 1, 128, 128, 128),
    (1, 4, 2, 72, 128, 128),
    (1, 1, 1, 64, 128, 1024),
    (2, 2, 1, 64, 128, 128),
    (1, 2, 1, 64, 128, 15),
    (1, 1, 1, 32, 128, 258),
    (1, 1, 1, 128, 256, 1024),
]


class BlackwellFMHAParityTest(unittest.TestCase):
    def _run_problems(self, deterministic):
        compared = 0
        for i, (b, hq, hkv, d, sq, skv) in enumerate(PROBLEMS):
            for is_causal in (False, True):
                reset_everything(seed=i, deterministic=deterministic)
                compared += run_fmha_case(
                    backend=BACKEND,
                    batch=b,
                    heads=hq,
                    heads_kv=hkv,
                    head_dim=d,
                    seqlen_q=sq,
                    seqlen_kv=skv,
                    is_causal=is_causal,
                    dtypes=DTYPES,
                    deterministic=deterministic,
                    configs_to_test=3,
                )
        if compared == 0:
            self.skipTest(
                "no launchable configs for this backend on this GPU/hub build"
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_default(self):
        self._run_problems(deterministic=False)

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_deterministic(self):
        self._run_problems(deterministic=True)

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_randsweep(self):
        compared = 0
        for i in range(12):
            reset_everything(seed=6000 + i, deterministic=False)
            p = random_fmha_problem(BACKEND)
            for is_causal in (False, True):
                compared += run_fmha_case(
                    backend=BACKEND,
                    is_causal=is_causal,
                    dtypes=DTYPES,
                    deterministic=False,
                    **p,
                )
        if compared == 0:
            self.skipTest(
                "no launchable configs for this backend on this GPU/hub build"
            )


if __name__ == "__main__":
    unittest.main()
