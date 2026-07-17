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

"""Performance regression tests: new profiler vs hard-coded old-profiler numbers.

For each hard-coded use case we build a Problem with the same settings the old
profiler was run with (see docs/profiler.md / docs/sample-outputs/*.txt),
profile forward pass up to N times, and pass as soon as ANY run's speedup
ratio falls inside the allowed band. If none of the N runs pass, fail with a
summary of observed min/max runtime and the expected band.

Convention (speedup ratio = expected_runtime_us / actual_runtime_us):
  - ratio > 1 : speedup (new is faster than old)
  - ratio < 1 : slowdown (new is slower than old)

Tolerance:
  - no worse than `MIN_SPEEDUP` slowdown (default 0.95 → 5% slower allowed)
  - no better than `MAX_SPEEDUP` speedup  (default 1.25 → up to 25% faster allowed)

Knobs (env vars, all optional):
  NATTEN_PERFTEST_PROFILES=<int>     max runs per case (default 20; early-exit on first pass)
  NATTEN_PERFTEST_WARMUPS=<int>      warmup steps per run (default 3)
  NATTEN_PERFTEST_SPEEDUP_MIN=<float> global slowdown floor (default 0.95, min 0.90)
  NATTEN_PERFTEST_SPEEDUP_MAX=<float> global speedup ceiling (default 1.25)

Per-case overrides: add `"min_speedup"` / `"max_speedup"` to the case dict.

To add/edit a case: edit the dict at the top of this file. `expected_runtime_us` comes from
the corresponding `docs/sample-outputs/profiler-*.txt` file (FnaForward /
fprop kernel runtime). Old profiler defaults to dtype=fp16.

Run commands (examples):
  # all cases on current GPU, defaults:
  pytest nattenprof/tests/test_perf_regression.py -v

  # single case in isolation (cleanest numbers — no in-suite GPU state):
  pytest nattenprof/tests/test_perf_regression.py::test_perf_regression_hopper[3d_hunyuan_gna_hopper_fna] -v

  # more runs + looser thresholds:
  NATTEN_PERFTEST_PROFILES=50 NATTEN_PERFTEST_SPEEDUP_MIN=0.90 pytest nattenprof/tests/test_perf_regression.py -v
"""

import os
from typing import Any, Dict

import pytest
import torch

from nattenprof.engine import ProfileOptions
from nattenprof.problem import NAProblem, PerfConfig, SdpaProblem
from nattenprof.tests._skipif import skip_no_blackwell, skip_no_hopper
from nattenprof.trace import KernelType

# ---- Tolerance + run knobs. Override via env vars (see module docstring). ----
MIN_SPEEDUP = float(os.environ.get("NATTEN_PERFTEST_SPEEDUP_MIN", "0.95"))
MAX_SPEEDUP = float(os.environ.get("NATTEN_PERFTEST_SPEEDUP_MAX", "1.25"))
RUNS_PER_CASE = int(os.environ.get("NATTEN_PERFTEST_PROFILES", "20"))
WARMUP_STEPS = int(os.environ.get("NATTEN_PERFTEST_WARMUPS", "3"))


# =============================================================================
# Hardcoded use cases + old-profiler forward-pass runtimes (microseconds).
# Pulled from docs/sample-outputs/profiler-*.txt (FnaForward / flash_fprop
# kernel lines). dtype=fp16 (old profiler default).
# =============================================================================


HOPPER_CASES: Dict[str, Dict[str, Any]] = {
    "1d_32k_cudnn": {
        "kind": "sdpa",
        "seqlen": 32768,
        "backend": "cudnn",
        "expected_runtime_us": 787.392,
    },
    "1d_32k_w2k_hopper_fna": {
        "kind": "na",
        "input_size": (32768,),
        "window_size": (2048,),
        "backend": "hopper-fna",
        "expected_runtime_us": 94.848,
    },
    "1d_32k_w2k_s256_hopper_fna": {
        "kind": "na",
        "input_size": (32768,),
        "window_size": (2048,),
        "stride": (256,),
        "backend": "hopper-fna",
        "expected_runtime_us": 62.687,
    },
    "1d_32k_w2k_s2k_hopper_fna": {
        "kind": "na",
        "input_size": (32768,),
        "window_size": (2048,),
        "stride": (2048,),
        "backend": "hopper-fna",
        "expected_runtime_us": 59.264,
    },
    "2d_flux_cudnn": {
        "kind": "sdpa",
        "seqlen": 256 * 256,
        "heads": 24,
        "backend": "cudnn",
        "expected_runtime_us": 97_435.0,
    },
    "2d_flux_na_hopper_fna": {
        "kind": "na",
        "input_size": (256, 256),
        "window_size": (80, 80),
        "heads": 24,
        "backend": "hopper-fna",
        "q_tile": (16, 8),
        "kv_tile": (16, 8),
        "expected_runtime_us": 16_201.0,
    },
    "2d_flux_gna_hopper_fna": {
        "kind": "na",
        "input_size": (256, 256),
        "window_size": (80, 80),
        "stride": (16, 16),
        "heads": 24,
        "backend": "hopper-fna",
        "q_tile": (16, 8),
        "kv_tile": (16, 8),
        "expected_runtime_us": 7_914.0,
    },
    "3d_hunyuan_cudnn": {
        "kind": "sdpa",
        "seqlen": 30 * 48 * 80,
        "heads": 24,
        "backend": "cudnn",
        "expected_runtime_us": 283_327.0,
    },
    "3d_hunyuan_na_hopper_fna": {
        "kind": "na",
        "input_size": (30, 48, 80),
        "window_size": (18, 24, 24),
        "heads": 24,
        "backend": "hopper-fna",
        "q_tile": (2, 8, 8),
        "kv_tile": (2, 8, 8),
        "expected_runtime_us": 79_043.0,
    },
    "3d_hunyuan_gna_hopper_fna": {
        "kind": "na",
        "input_size": (30, 48, 80),
        "window_size": (18, 24, 24),
        "stride": (16, 8, 8),
        "heads": 24,
        "backend": "hopper-fna",
        "q_tile": (2, 8, 8),
        "kv_tile": (2, 8, 8),
        "expected_runtime_us": 23_359.0,
    },
}


BLACKWELL_CASES: Dict[str, Dict[str, Any]] = {
    "1d_32k_cudnn": {
        "kind": "sdpa",
        "seqlen": 32768,
        "backend": "cudnn",
        "expected_runtime_us": 424.766,
    },
    "1d_32k_w2k_blackwell_fna": {
        "kind": "na",
        "input_size": (32768,),
        "window_size": (2048,),
        "backend": "blackwell-fna",
        "expected_runtime_us": 58.111,
    },
    "1d_32k_w2k_s256_blackwell_fna": {
        "kind": "na",
        "input_size": (32768,),
        "window_size": (2048,),
        "stride": (256,),
        "backend": "blackwell-fna",
        "expected_runtime_us": 37.888,
    },
    "1d_32k_w2k_s2k_blackwell_fna": {
        "kind": "na",
        "input_size": (32768,),
        "window_size": (2048,),
        "stride": (2048,),
        "backend": "blackwell-fna",
        "expected_runtime_us": 38.240,
    },
    "2d_flux_cudnn": {
        "kind": "sdpa",
        "seqlen": 256 * 256,
        "heads": 24,
        "backend": "cudnn",
        "expected_runtime_us": 43_313.0,
    },
    "2d_flux_na_blackwell_fna": {
        "kind": "na",
        "input_size": (256, 256),
        "window_size": (80, 80),
        "heads": 24,
        "backend": "blackwell-fna",
        "q_tile": (16, 16),
        "kv_tile": (16, 8),
        "expected_runtime_us": 8_692.0,
    },
    "2d_flux_gna_blackwell_fna": {
        "kind": "na",
        "input_size": (256, 256),
        "window_size": (80, 80),
        "stride": (16, 16),
        "heads": 24,
        "backend": "blackwell-fna",
        "q_tile": (16, 16),
        "kv_tile": (16, 8),
        "expected_runtime_us": 4_114.0,
    },
    "3d_hunyuan_cudnn": {
        "kind": "sdpa",
        "seqlen": 30 * 48 * 80,
        "heads": 24,
        "backend": "cudnn",
        "expected_runtime_us": 135_265.0,
    },
    "3d_hunyuan_na_blackwell_fna": {
        "kind": "na",
        "input_size": (30, 48, 80),
        "window_size": (18, 24, 24),
        "heads": 24,
        "backend": "blackwell-fna",
        "q_tile": (4, 8, 8),
        "kv_tile": (2, 8, 8),
        "expected_runtime_us": 42_243.0,
    },
    "3d_hunyuan_gna_blackwell_fna": {
        "kind": "na",
        "input_size": (30, 48, 80),
        "window_size": (18, 24, 24),
        "stride": (16, 8, 8),
        "heads": 24,
        "backend": "blackwell-fna",
        "q_tile": (4, 8, 8),
        "kv_tile": (2, 8, 8),
        "expected_runtime_us": 13_010.0,
    },
}


# =============================================================================
# Helpers
# =============================================================================


def _build_problem(cfg: Dict[str, Any]):
    import math

    dtype = torch.float16
    heads = cfg.get("heads", 1)
    dim = cfg.get("dim", 128)
    if cfg["kind"] == "sdpa":
        perf = PerfConfig(fmha_backend=cfg["backend"])
        return SdpaProblem(
            batch_size=1,
            heads=heads,
            heads_kv=heads,
            dim=dim,
            dim_value=dim,
            dtype=dtype,
            bwd=False,
            perf=perf,
            seqlen_q=cfg["seqlen"],
            seqlen_kv=cfg["seqlen"],
            is_causal=False,
        )
    if cfg["kind"] == "na":
        q_shape = cfg.get("q_tile")
        kv_shape = cfg.get("kv_tile")
        perf = PerfConfig(
            fna_backend=cfg["backend"],
            q_tile_shape=q_shape,
            kv_tile_shape=kv_shape,
            q_tile_size=math.prod(q_shape) if q_shape else None,
            kv_tile_size=math.prod(kv_shape) if kv_shape else None,
        )
        input_size = cfg["input_size"]
        return NAProblem(
            batch_size=1,
            heads=heads,
            heads_kv=heads,
            dim=dim,
            dim_value=dim,
            dtype=dtype,
            bwd=False,
            perf=perf,
            input_size=input_size,
            window_size=cfg["window_size"],
            stride=cfg.get("stride", tuple(1 for _ in input_size)),
            dilation=tuple(1 for _ in input_size),
            is_causal=tuple(False for _ in input_size),
        )
    raise ValueError(f"unknown kind: {cfg['kind']}")


def _forward_us(result) -> float:
    return sum(
        k.time_us
        for k in result.kernels
        if k.kernel_type == KernelType.AttentionForward
    )


def _check_performance(cfg: Dict[str, Any]) -> None:
    if cfg["kind"] == "na":
        from natten import set_memory_usage_preference, use_kv_parallelism_in_fused_na

        use_kv_parallelism_in_fused_na(True)
        set_memory_usage_preference("unrestricted")

    settings = ProfileOptions(
        warmup_steps=WARMUP_STEPS, init_mode="randn", memory_limit=10.0, seed=42
    )

    expected_runtime_us = cfg["expected_runtime_us"]
    lo = cfg.get("min_speedup", MIN_SPEEDUP)
    hi = cfg.get("max_speedup", MAX_SPEEDUP)
    assert lo >= 0.90, "configured min_speedup below 0.90 (>10% slowdown) — stop."

    runs = []
    for _ in range(RUNS_PER_CASE):
        problem = _build_problem(cfg)
        problem.check_config()
        us = _forward_us(problem.profile(settings))
        runs.append(us)
        ratio = expected_runtime_us / us
        if lo <= ratio <= hi:
            return  # early exit: one run satisfied the band

    # None of the runs landed in the band.
    min_us = min(runs)
    max_us = max(runs)
    best_ratio = expected_runtime_us / min_us  # highest speedup across runs
    worst_ratio = expected_runtime_us / max_us  # lowest speedup across runs
    raise AssertionError(
        f"[{cfg['name']}] no run in {RUNS_PER_CASE} satisfied the band.\n"
        f"  expected_runtime_us               = {expected_runtime_us:.3f}\n"
        f"  actual_runtime_us  min          = {min_us:.3f}  (ratio {best_ratio:.3f})\n"
        f"  actual_runtime_us  max          = {max_us:.3f}  (ratio {worst_ratio:.3f})\n"
        f"  allowed speedup band = [{lo}, {hi}]\n"
        f"  equivalent actual_runtime_us    = [{expected_runtime_us / hi:.3f}, {expected_runtime_us / lo:.3f}]"
    )


# =============================================================================
# Tests
# =============================================================================


@skip_no_hopper
@pytest.mark.parametrize("name", list(HOPPER_CASES.keys()))
def test_perf_regression_hopper(name: str):
    cfg = {**HOPPER_CASES[name], "name": name}
    _check_performance(cfg)


@skip_no_blackwell
@pytest.mark.parametrize("name", list(BLACKWELL_CASES.keys()))
def test_perf_regression_blackwell(name: str):
    cfg = {**BLACKWELL_CASES[name], "name": name}
    _check_performance(cfg)
