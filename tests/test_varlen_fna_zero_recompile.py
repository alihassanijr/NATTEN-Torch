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

MAX_SEQLEN = 2**15  # 32768

# Timeout for the very first run (initial torch.compile compilation)
FIRST_RUN_TIMEOUT_S = 5.0
# Timeout for all subsequent runs (zero recompiles expected)
RERUN_TIMEOUT_S = 0.1

# When True, ensure per-sequence kernel sizes are not all identical within a use case.
FORCE_VAR_PARAM = False
# When True, vary stride per sequence (random choice from [1, 2]).
VARY_STRIDE = True
# When True, vary dilation per sequence (random valid value).
VARY_DILATION = True


def _generate_use_case():
    """Generate one random use case with variable layouts and per-sequence params.

    All use cases share a fixed QKV shape (MAX_SEQLEN), so torch.compile's
    dynamic mode should never recompile.
    """
    num_seqs = random.randint(2, 6)
    token_layout_list = []
    kernel_size_list = []
    stride_list = []
    dilation_list = []

    for _ in range(num_seqs):
        h = random.choice([16, 24, 32, 40, 48])
        w = random.choice([16, 24, 32, 40, 48])
        token_layout_list.append((h, w))

        ks = random.choice([k for k in [3, 5, 7] if k <= min(h, w)])
        kernel_size_list.append((ks, ks))

        if VARY_STRIDE:
            sh = random.choice([s for s in [1, 2] if s <= h])
            sw = random.choice([s for s in [1, 2] if s <= w])
            stride_list.append((sh, sw))
        else:
            stride_list.append((1, 1))

        if VARY_DILATION:
            # (ks - 1) * d + 1 <= dim  =>  d <= (dim - 1) / (ks - 1)
            max_dh = (h - 1) // (ks - 1) if ks > 1 else 1
            max_dw = (w - 1) // (ks - 1) if ks > 1 else 1
            dh = random.randint(1, max(1, min(max_dh, 3)))
            dw = random.randint(1, max(1, min(max_dw, 3)))
            dilation_list.append((dh, dw))
        else:
            dilation_list.append((1, 1))

    if FORCE_VAR_PARAM and len(set(kernel_size_list)) == 1 and num_seqs > 1:
        # Regenerate last entry with a different kernel size
        alt_choices = [
            k
            for k in [3, 5, 7]
            if k <= min(token_layout_list[-1]) and (k, k) != kernel_size_list[0]
        ]
        if alt_choices:
            ks = random.choice(alt_choices)
            kernel_size_list[-1] = (ks, ks)

    total_seqlen = sum(math.prod(x) for x in token_layout_list)
    assert (
        total_seqlen <= MAX_SEQLEN
    ), f"total_seqlen={total_seqlen} exceeds MAX_SEQLEN={MAX_SEQLEN}"

    return {
        "token_layout_list": token_layout_list,
        "kernel_size_list": kernel_size_list,
        "stride_list": stride_list,
        "dilation_list": dilation_list,
        "total_seqlen": total_seqlen,
    }


def _make_qkv(heads, head_dim):
    shape = (1, MAX_SEQLEN, heads, head_dim)
    q = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    return q, k, v


def _prepare_inputs(use_case, heads, head_dim, backend):
    """Generate fresh QKV (always MAX_SEQLEN) and fresh metadata for a use case."""
    q, k, v = _make_qkv(heads, head_dim)
    metadata = configure_varlen(
        token_layout_list=use_case["token_layout_list"],
        head_dim=head_dim,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        requires_grad=True,
        kernel_size_list=use_case["kernel_size_list"],
        stride_list=use_case["stride_list"],
        dilation_list=use_case["dilation_list"],
        backend=backend,
    )
    return q, k, v, metadata


def _attention_fn(q, k, v, metadata):
    return neighborhood_attention_varlen(q, k, v, metadata=metadata)


def _run_zero_recompile_check(backend, num_use_cases, runs_per_use_case):
    random.seed(42)
    torch.manual_seed(42)
    torch.compiler.reset()
    torch._dynamo.config.cache_size_limit = 1
    torch._dynamo.config.accumulated_recompile_limit = 1
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch.cuda.empty_cache()

    heads = random.choice([1, 2, 4, 8, 16])
    head_dim = random.choice([32, 64, 128])
    use_cases = [_generate_use_case() for _ in range(num_use_cases)]
    total_runs = num_use_cases * runs_per_use_case

    logger.debug(
        f"=== {backend}: {num_use_cases} use cases x {runs_per_use_case} runs "
        f"= {total_runs} iterations, zero recompiles expected ==="
    )
    logger.debug(f"  heads={heads}, head_dim={head_dim}, MAX_SEQLEN={MAX_SEQLEN}")
    for i, uc in enumerate(use_cases):
        logger.debug(
            f"  uc[{i}]: layouts={uc['token_layout_list']}, "
            f"total={uc['total_seqlen']}, "
            f"ks={uc['kernel_size_list']}, "
            f"stride={uc['stride_list']}, "
            f"dilation={uc['dilation_list']}"
        )

    fn = torch.compile(_attention_fn, fullgraph=True, dynamic=True)
    run_counts = [0] * num_use_cases
    first_run_done = False

    for iteration in range(total_runs):
        eligible = [
            i for i in range(num_use_cases) if run_counts[i] < runs_per_use_case
        ]
        uc_idx = random.choice(eligible)
        run_counts[uc_idx] += 1
        run_num = run_counts[uc_idx]

        q, k, v, metadata = _prepare_inputs(use_cases[uc_idx], heads, head_dim, backend)

        logger.debug(
            f"  [iter {iteration:3d}] uc={uc_idx} "
            f"run={run_num}/{runs_per_use_case} "
            f"layouts={use_cases[uc_idx]['token_layout_list']} "
            f"ks={use_cases[uc_idx]['kernel_size_list']} "
            f"stride={use_cases[uc_idx]['stride_list']} "
            f"dilation={use_cases[uc_idx]['dilation_list']} ..."
        )

        torch.cuda.synchronize()
        t0 = time.time()
        out = fn(q, k, v, metadata)
        out.sum().backward()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        if not first_run_done:
            first_run_done = True
            timeout = FIRST_RUN_TIMEOUT_S
            tag = "COMPILE"
        else:
            timeout = RERUN_TIMEOUT_S
            tag = "cached "

        logger.debug(
            f"  [iter {iteration:3d}] uc={uc_idx} "
            f"run={run_num}/{runs_per_use_case} "
            f"{tag} {elapsed:.3f}s (limit={timeout:.1f}s)"
        )

        assert elapsed < timeout, (
            f"use_case={uc_idx} run={run_num} took {elapsed:.3f}s, "
            f"exceeds {'compile' if tag == 'COMPILE' else 'cached'} "
            f"limit of {timeout:.1f}s"
            + ("" if tag == "COMPILE" else " (likely unexpected recompile)")
        )

    logger.debug(f"=== PASSED {backend} ===")


class VarlenFNAZeroRecompileTest(unittest.TestCase):
    # Hopper FNA

    @skip_if_hopper_kernels_not_supported()
    def test_hopper_fna(self):
        _run_zero_recompile_check("hopper-fna", num_use_cases=3, runs_per_use_case=4)

    # Blackwell FNA

    @skip_if_blackwell_kernels_not_supported()
    def test_blackwell_fna(self):
        _run_zero_recompile_check(
            "blackwell-fna", num_use_cases=20, runs_per_use_case=6
        )
