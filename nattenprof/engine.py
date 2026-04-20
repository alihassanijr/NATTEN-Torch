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

"""Profiling engine: torch profiler capture, warmup, measurement, and
high-level profile_* functions used by CLI and batch mode."""

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity

from nattenprof.ops import run_attn, run_na, run_sdpa
from nattenprof.output import KernelResult, ProfileResult
from nattenprof.problem import AttentionProblem, NAProblem
from nattenprof.tensors import InitMode, TensorPool
from nattenprof.trace import convert_trace_to_results

IS_CUDA = torch.cuda.is_available()
_PROFILER_ACTIVITY = ProfilerActivity.CUDA if IS_CUDA else ProfilerActivity.CPU

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}


# Tile values: NA uses tuples (q_tile_shape), attn/sdpa uses ints (q_tile_size).
TileValue = Optional[Union[int, Tuple[int, ...]]]


@dataclass
class ProfileOptions:
    """Profiling knobs shared by profile_na / profile_attn / profile_sdpa.

    Not every op uses every field. For example, profile_sdpa ignores backend-perf
    fields like persistent and schedule.
    """

    # Backend selection
    backend: Optional[str] = None
    fmha_backend: Optional[str] = None  # NA only
    # Tile config
    q_tile: TileValue = None
    kv_tile: TileValue = None
    bwd_q_tile: TileValue = None
    bwd_kv_tile: TileValue = None
    # Kernel perf
    persistent: bool = False
    schedule: Optional[str] = None
    compile: bool = False
    # Profiling
    bwd: bool = False
    warmup_steps: int = 10
    # TensorPool
    init_mode: str = "randn"
    memory_limit: float = 10.0
    seed: int = 42


# ---- Low-level profiling primitives ----


def profile_op(
    pool: TensorPool,
    run_fn: Callable[[Dict[str, Tensor]], None],
    warmup_steps: int = 10,
    max_retries: int = 5,
) -> Optional[List[KernelResult]]:
    """Run warmup, then profile a single iteration with torch profiler.

    Warmup runs are also wrapped in torch_profile so the profiler itself is warm
    by the time we capture the real trace.
    """
    for _ in range(warmup_steps):
        tensors = pool.get()
        if IS_CUDA:
            torch.cuda.synchronize()
        with torch_profile(activities=[_PROFILER_ACTIVITY]):
            run_fn(tensors)
            if IS_CUDA:
                torch.cuda.synchronize()

    for _attempt in range(max_retries):
        tensors = pool.get()
        if IS_CUDA:
            torch.cuda.synchronize()

        with torch_profile(activities=[_PROFILER_ACTIVITY]) as prof:
            run_fn(tensors)
            if IS_CUDA:
                torch.cuda.synchronize()

        results = convert_trace_to_results(prof)

        if results is not None and len(results) > 0:
            return results

        if IS_CUDA:
            torch.cuda.synchronize()
        time.sleep(0.5)
        if IS_CUDA:
            torch.cuda.synchronize()

    return None


def measure_wall_time_ms(
    pool: TensorPool,
    run_fn: Callable[[Dict[str, Tensor]], None],
    warmup_steps: int = 5,
) -> float:
    """Measure end-to-end wall time (all kernels + CUDA overhead) in milliseconds.

    Uses cuda events, no profiler. Used by the optimize loop where we don't need
    per-kernel breakdown — just total runtime to pick the fastest config.
    """
    for _ in range(warmup_steps):
        tensors = pool.get()
        run_fn(tensors)

    tensors = pool.get()

    if IS_CUDA:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    run_fn(tensors)

    if IS_CUDA:
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    else:
        return (time.time() - start_time) * 1e3


# ---- High-level profile functions ----


def _profiling_info_str(warmup_steps: int) -> str:
    return f"  warmup_steps={warmup_steps}, profiling_steps=1"


def _make_pool(problem, opts: ProfileOptions, heads_last: bool = True) -> TensorPool:
    disable_backward = not opts.bwd
    if isinstance(problem, AttentionProblem):
        shapes = problem.get_tensor_shapes(heads_last=heads_last)
    else:
        shapes = problem.get_tensor_shapes()
    return TensorPool(
        shapes=shapes,
        dtype=problem.dtype,
        device=torch.device("cuda"),
        init_mode=InitMode(opts.init_mode),
        memory_limit_gb=opts.memory_limit,
        seed=opts.seed,
        requires_grad=not disable_backward,
    )


def _format_tile_info(opts: ProfileOptions) -> str:
    """Format tile shape info for display in use case header."""
    parts = []
    if opts.q_tile is not None:
        parts.append(f"q_tile={opts.q_tile}")
    if opts.kv_tile is not None:
        parts.append(f"kv_tile={opts.kv_tile}")
    if opts.bwd_q_tile is not None:
        parts.append(f"bwd_q_tile={opts.bwd_q_tile}")
    if opts.bwd_kv_tile is not None:
        parts.append(f"bwd_kv_tile={opts.bwd_kv_tile}")
    if not parts:
        return ""
    return "  " + ", ".join(parts)


def _build_result(
    operation: str,
    problem,
    opts: ProfileOptions,
    kernels: List[KernelResult],
    config: Dict[str, Any],
) -> ProfileResult:
    if operation == "na":
        use_case = problem.format_use_case(
            backend=opts.backend, fmha_backend=opts.fmha_backend
        )
    else:
        use_case = problem.format_use_case(backend=opts.backend)
    tile_info = _format_tile_info(opts)
    if tile_info:
        use_case += "\n" + tile_info
    use_case += "\n" + _profiling_info_str(opts.warmup_steps)

    return ProfileResult(
        operation=operation, config=config, kernels=kernels, use_case_str=use_case
    )


def profile_na(
    problem: NAProblem, opts: Optional[ProfileOptions] = None
) -> ProfileResult:
    """Profile neighborhood attention."""
    from natten import (
        allow_flex_compile,
        set_memory_usage_preference,
        use_kv_parallelism_in_fused_na,
    )

    opts = opts or ProfileOptions()

    if opts.compile:
        allow_flex_compile(True, True)
    use_kv_parallelism_in_fused_na(True)
    set_memory_usage_preference("unrestricted")

    disable_backward = not opts.bwd
    torch.set_grad_enabled(not disable_backward)

    pool = _make_pool(problem, opts)

    run_fn = partial(
        run_na,
        problem=problem,
        backend=opts.backend,
        fmha_backend=opts.fmha_backend,
        q_tile_shape=opts.q_tile,
        kv_tile_shape=opts.kv_tile,
        backward_q_tile_shape=opts.bwd_q_tile,
        backward_kv_tile_shape=opts.bwd_kv_tile,
        run_persistent_kernel=opts.persistent,
        kernel_schedule=opts.schedule,
        torch_compile=opts.compile,
        disable_backward=disable_backward,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=opts.warmup_steps)

    if results is None or len(results) == 0:
        raise RuntimeError("Profiler captured no kernel events after retries.")

    config = {
        "input_size": list(problem.input_size),
        "window_size": list(problem.window_size),
        "stride": list(problem.stride),
        "dilation": list(problem.dilation),
        "is_causal": list(problem.is_causal),
        "batch_size": problem.batch_size,
        "heads": problem.heads,
        "heads_kv": problem.heads_kv,
        "dim": problem.dim,
        "dim_value": problem.dim_value,
        "dtype": str(problem.dtype),
        "backend": opts.backend,
        "fmha_backend": opts.fmha_backend,
    }

    return _build_result("na", problem, opts, results, config)


def profile_attn(
    problem: AttentionProblem, opts: Optional[ProfileOptions] = None
) -> ProfileResult:
    """Profile NATTEN standard attention."""
    opts = opts or ProfileOptions()

    disable_backward = not opts.bwd
    torch.set_grad_enabled(not disable_backward)

    pool = _make_pool(problem, opts, heads_last=True)

    run_fn = partial(
        run_attn,
        problem=problem,
        backend=opts.backend,
        q_tile_size=opts.q_tile,
        kv_tile_size=opts.kv_tile,
        backward_q_tile_size=opts.bwd_q_tile,
        backward_kv_tile_size=opts.bwd_kv_tile,
        run_persistent_kernel=opts.persistent,
        kernel_schedule=opts.schedule,
        torch_compile=opts.compile,
        disable_backward=disable_backward,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=opts.warmup_steps)

    if results is None or len(results) == 0:
        raise RuntimeError("Profiler captured no kernel events after retries.")

    config = {
        "seqlen_q": problem.seqlen_q,
        "seqlen_kv": problem.seqlen_kv,
        "batch_size": problem.batch_size,
        "heads": problem.heads,
        "heads_kv": problem.heads_kv,
        "dim": problem.dim,
        "dim_value": problem.dim_value,
        "dtype": str(problem.dtype),
        "is_causal": problem.is_causal,
        "varlen": problem.is_varlen,
        "backend": opts.backend,
    }

    return _build_result("attn", problem, opts, results, config)


def profile_sdpa(
    problem: AttentionProblem, opts: Optional[ProfileOptions] = None
) -> ProfileResult:
    """Profile torch SDPA baseline."""
    opts = opts or ProfileOptions(backend="cudnn")
    if opts.backend is None:
        opts.backend = "cudnn"

    disable_backward = not opts.bwd
    torch.set_grad_enabled(not disable_backward)

    pool = _make_pool(problem, opts, heads_last=False)

    run_fn = partial(
        run_sdpa,
        backend=opts.backend,
        is_causal=problem.is_causal,
        disable_backward=disable_backward,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=opts.warmup_steps)

    if results is None or len(results) == 0:
        raise RuntimeError("Profiler captured no kernel events after retries.")

    config = {
        "seqlen_q": problem.seqlen_q,
        "seqlen_kv": problem.seqlen_kv,
        "batch_size": problem.batch_size,
        "heads": problem.heads,
        "heads_kv": problem.heads_kv,
        "dim": problem.dim,
        "dim_value": problem.dim_value,
        "dtype": str(problem.dtype),
        "is_causal": problem.is_causal,
        "backend": opts.backend,
    }

    return _build_result("sdpa", problem, opts, results, config)
