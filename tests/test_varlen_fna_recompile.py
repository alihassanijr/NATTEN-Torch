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

import math
import random
import time
import unittest

import torch
from natten.utils import log
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_hopper_kernels_not_supported,
)
from natten.varlen import configure_varlen, neighborhood_attention_varlen

logger = log.get_logger(__name__)

DEVICE = "cuda"
DTYPE = torch.float16

# Timeout for first run of each use case (includes torch.compile compilation)
FIRST_RUN_TIMEOUT_S = 10.0
# Timeout for subsequent runs (no recompilation expected)
RERUN_TIMEOUT_S = 0.1


def _generate_use_case():
    """Generate one random use case with arbitrary shapes and params."""
    num_seqs = random.randint(2, 6)
    token_layout_list = [
        (random.choice([16, 32, 48, 64]), random.choice([16, 32, 48, 64]))
        for _ in range(num_seqs)
    ]
    kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
    total_seqlen = sum(math.prod(x) for x in token_layout_list)
    return {
        "token_layout_list": token_layout_list,
        "kernel_size": kernel_size,
        "total_seqlen": total_seqlen,
    }


def _make_qkv(total_seqlen, heads, head_dim):
    shape = (1, total_seqlen, heads, head_dim)
    q = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    return q, k, v


def _prepare_inputs(use_case, heads, head_dim, backend):
    """Generate fresh QKV tensors and fresh metadata for a use case."""
    q, k, v = _make_qkv(use_case["total_seqlen"], heads, head_dim)
    metadata = configure_varlen(
        token_layout_list=use_case["token_layout_list"],
        head_dim=head_dim,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        requires_grad=True,
        kernel_size=use_case["kernel_size"],
        backend=backend,
    )
    return q, k, v, metadata


def _attention_fn(q, k, v, metadata):
    return neighborhood_attention_varlen(q, k, v, metadata=metadata)


def _run_recompile_check(backend, num_use_cases, runs_per_use_case):
    random.seed(42)
    torch.manual_seed(42)
    torch.compiler.reset()
    torch._dynamo.config.cache_size_limit = num_use_cases
    torch._dynamo.config.accumulated_recompile_limit = num_use_cases
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch.cuda.empty_cache()

    heads = random.choice([1, 2, 4, 8, 16])
    head_dim = random.choice([32, 64, 128])
    use_cases = [_generate_use_case() for _ in range(num_use_cases)]
    total_runs = num_use_cases * runs_per_use_case

    logger.debug(
        f"=== {backend}: {num_use_cases} use cases x {runs_per_use_case} runs "
        f"= {total_runs} total iterations ==="
    )
    logger.debug(f"  heads={heads}, head_dim={head_dim}")
    for i, uc in enumerate(use_cases):
        logger.debug(
            f"  uc[{i}]: total_seqlen={uc['total_seqlen']}, "
            f"layouts={uc['token_layout_list']}, "
            f"kernel_size={uc['kernel_size']}"
        )

    fn = torch.compile(_attention_fn, fullgraph=True, dynamic=True)
    run_counts = [0] * num_use_cases

    for iteration in range(total_runs):
        # Pick a random use case that hasn't exhausted its runs
        eligible = [
            i for i in range(num_use_cases) if run_counts[i] < runs_per_use_case
        ]
        uc_idx = random.choice(eligible)
        run_counts[uc_idx] += 1
        is_first = run_counts[uc_idx] == 1
        run_num = run_counts[uc_idx]

        q, k, v, metadata = _prepare_inputs(use_cases[uc_idx], heads, head_dim, backend)

        torch.cuda.synchronize()
        t0 = time.time()
        out = fn(q, k, v, metadata)
        out.sum().backward()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        timeout = FIRST_RUN_TIMEOUT_S if is_first else RERUN_TIMEOUT_S
        tag = "COMPILE" if is_first else "cached "

        logger.debug(
            f"  [iter {iteration:3d}] uc={uc_idx} "
            f"run={run_num}/{runs_per_use_case} "
            f"{tag} {elapsed:.3f}s (limit={timeout:.1f}s)"
        )

        assert elapsed < timeout, (
            f"use_case={uc_idx} run={run_num} took {elapsed:.3f}s, "
            f"exceeds {'compile' if is_first else 'cached'} limit of {timeout:.1f}s"
            + ("" if is_first else " (likely unexpected recompile)")
        )

    logger.debug(f"=== PASSED {backend} ===")


class VarlenFNARecompileTest(unittest.TestCase):
    # Hopper FNA

    @skip_if_hopper_kernels_not_supported()
    def test_hopper_fna(self):
        _run_recompile_check("hopper-fna", num_use_cases=3, runs_per_use_case=4)

    # Blackwell FNA

    @skip_if_blackwell_kernels_not_supported()
    def test_blackwell_fna(self):
        _run_recompile_check("blackwell-fna", num_use_cases=20, runs_per_use_case=6)
