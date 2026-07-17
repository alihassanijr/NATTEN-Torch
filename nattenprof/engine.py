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

"""Profiling engine primitives.

Compute + perf knobs live on Problem. This file only hosts profiling-side
utilities: torch profiler capture loop, wall-time measurement, and the
ProfileOptions dataclass for pure profiling knobs.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity

from nattenprof.output import KernelResult
from nattenprof.tensors import TensorPool
from nattenprof.trace import convert_trace_to_results

IS_CUDA = torch.cuda.is_available()
_PROFILER_ACTIVITY = ProfilerActivity.CUDA if IS_CUDA else ProfilerActivity.CPU


@dataclass
class ProfileOptions:
    """Profiling-level knobs (nothing about the computation itself)."""

    warmup_steps: int = 10
    init_mode: str = "randn"
    memory_limit: float = 10.0
    seed: int = 42


def profile_op(
    pool: TensorPool,
    run_fn: Callable[[Dict[str, Tensor]], None],
    warmup_steps: int = 10,
    max_retries: int = 5,
) -> Optional[List[KernelResult]]:
    """Warmup + torch profiler capture of a single iteration.

    Warmup is wrapped in torch_profile too so the profiler itself is warm
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
    """End-to-end wall time (all kernels + CUDA overhead) in ms via cuda events.

    Used by the optimize loop where per-kernel breakdown isn't needed —
    just total runtime for picking the fastest config.
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
    return (time.time() - start_time) * 1e3
