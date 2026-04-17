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

"""Parity: old natten.profiler vs new nattenprof.

Both profilers are invoked via their Python APIs in the same process — no
subprocesses, no text parsing, no /tmp files. For each parametrized config
we capture a raw torch profiler trace from each side and compare:

  - raw kernel symbols          : IDENTICAL set
  - num_calls per symbol        : IDENTICAL
  - time_us per symbol          : within TIME_TOLERANCE_RATIO
  - total time_us               : within TIME_TOLERANCE_RATIO
  - canonical op classification : IDENTICAL after normalization
    (TokPerm + TokUnperm collapse into old TokenPermute)

The canonical check is only meaningful for configs whose kernels the OLD
profiler actually recognizes (natten NA + FMHA). cuDNN/FAv2 symbols are
unknown to the old classifier, so we compare only raw traces for those.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch

# --- Old profiler (private APIs) ---
from natten.profiling_utils.formatting import (
    convert_to_natten_profiler_ops,
    get_natten_op,
    LibNattenOp,
)
from natten.profiling_utils.problem import Problem as OldProblem
from natten.profiling_utils.profiling import (
    _profile_fmha_with_torch,
    _profile_na_with_torch,
)
from natten.utils.device import get_device_cc
from torch.profiler import profile as torch_profile, ProfilerActivity

# --- New profiler ---
from nattenprof.ops import run_attn, run_na, run_sdpa
from nattenprof.problem import AttentionProblem, NAProblem
from nattenprof.tensors import InitMode, TensorPool
from nattenprof.trace import KernelType

HAS_CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _get_cc() -> int:
    return get_device_cc() if HAS_CUDA else 0


skip_no_hopper = pytest.mark.skipif(
    not HAS_CUDA or _get_cc() != 90,
    reason="Hopper kernels require SM90",
)

skip_no_blackwell = pytest.mark.skipif(
    not HAS_CUDA or _get_cc() not in (100, 103),
    reason="Blackwell kernels require SM100 or SM103",
)


# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------

# Timing comparison uses BOTH an absolute noise floor and a relative ratio.
# A kernel passes if EITHER |old - new| <= TIME_NOISE_FLOOR_US
# OR max/min <= TIME_TOLERANCE_RATIO. The absolute floor handles CUDA-timer
# quantization on sub-10us kernels; the ratio handles everything else.
TIME_TOLERANCE_RATIO = 1.15  # ±15%
TIME_NOISE_FLOOR_US = 1.5
WARMUP_STEPS = 10

# Skip tiny kernels entirely — both sides report sub-TIME_IGNORE_US. At that
# scale torch profiler timing is essentially ±1 tick of the GPU clock.
TIME_IGNORE_US = 0.5


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class ParityConfig:
    """One parity test case.

    op: "sdpa" | "attn" | "na"
    backend: profiler backend name (same string for old and new)
    For NA: input_size, window_size, stride, dilation, is_causal
    For attn/sdpa: seqlen
    """

    name: str
    op: str  # "sdpa" | "attn" | "na"
    dim: int = 128
    heads: int = 1
    heads_kv: int = 1
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16
    backend: Optional[str] = None
    bwd: bool = False

    # NA-only
    input_size: Optional[Tuple[int, ...]] = None
    window_size: Optional[Tuple[int, ...]] = None
    stride: Optional[Tuple[int, ...]] = None
    dilation: Optional[Tuple[int, ...]] = None
    is_causal: Optional[Tuple[bool, ...]] = None

    # attn/sdpa-only
    seqlen: Optional[int] = None

    # arch gating
    hopper_only: bool = False
    blackwell_only: bool = False


def _na_cfg(
    name: str,
    input_size: Tuple[int, ...],
    window_size: Tuple[int, ...],
    backend: str,
    bwd: bool = False,
    stride: Optional[Tuple[int, ...]] = None,
    hopper_only: bool = False,
    blackwell_only: bool = False,
) -> ParityConfig:
    ndim = len(input_size)
    return ParityConfig(
        name=name,
        op="na",
        backend=backend,
        bwd=bwd,
        input_size=input_size,
        window_size=window_size,
        stride=stride or tuple([1] * ndim),
        dilation=tuple([1] * ndim),
        is_causal=tuple([False] * ndim),
        hopper_only=hopper_only,
        blackwell_only=blackwell_only,
    )


def _attn_cfg(
    name: str,
    seqlen: int,
    backend: str,
    bwd: bool = False,
    hopper_only: bool = False,
    blackwell_only: bool = False,
) -> ParityConfig:
    return ParityConfig(
        name=name,
        op="attn",
        backend=backend,
        bwd=bwd,
        seqlen=seqlen,
        hopper_only=hopper_only,
        blackwell_only=blackwell_only,
    )


def _sdpa_cfg(
    name: str,
    seqlen: int,
    backend: str,
    dtype: torch.dtype = torch.bfloat16,
    bwd: bool = False,
) -> ParityConfig:
    return ParityConfig(
        name=name,
        op="sdpa",
        backend=backend,
        bwd=bwd,
        seqlen=seqlen,
        dtype=dtype,
    )


CONFIGS: List[ParityConfig] = [
    # SDPA — old classifier only marks everything "Unrecognized"; raw-only parity.
    _sdpa_cfg("sdpa_cudnn_fwd", 1024, "cudnn"),
    _sdpa_cfg("sdpa_cudnn_bwd", 1024, "cudnn", bwd=True),
    _sdpa_cfg("sdpa_fav2_fwd", 1024, "fav2", dtype=torch.float16),
    # NATTEN attention (hopper-fmha, cutlass-fmha)
    _attn_cfg("attn_hopper_fmha_fwd", 1024, "hopper-fmha", hopper_only=True),
    _attn_cfg("attn_hopper_fmha_bwd", 1024, "hopper-fmha", bwd=True, hopper_only=True),
    _attn_cfg("attn_cutlass_fmha_fwd", 1024, "cutlass-fmha"),
    _attn_cfg("attn_cutlass_fmha_bwd", 1024, "cutlass-fmha", bwd=True),
    # Neighborhood attention
    _na_cfg("na_1d_hopper_fna_fwd", (1024,), (256,), "hopper-fna", hopper_only=True),
    _na_cfg(
        "na_1d_hopper_fna_bwd",
        (1024,),
        (256,),
        "hopper-fna",
        bwd=True,
        hopper_only=True,
    ),
    _na_cfg("na_1d_cutlass_fna_fwd", (1024,), (256,), "cutlass-fna"),
    _na_cfg("na_1d_cutlass_fna_bwd", (1024,), (256,), "cutlass-fna", bwd=True),
    _na_cfg("na_2d_hopper_fna", (64, 64), (32, 32), "hopper-fna", hopper_only=True),
    _na_cfg(
        "na_3d_hopper_fna", (16, 16, 16), (8, 8, 8), "hopper-fna", hopper_only=True
    ),
    _na_cfg(
        "na_1d_hopper_strided",
        (1024,),
        (256,),
        "hopper-fna",
        stride=(128,),
        hopper_only=True,
    ),
    # Blackwell (SM100/SM103)
    _attn_cfg("attn_blackwell_fmha_fwd", 1024, "blackwell-fmha", blackwell_only=True),
    _attn_cfg(
        "attn_blackwell_fmha_bwd",
        1024,
        "blackwell-fmha",
        bwd=True,
        blackwell_only=True,
    ),
    _na_cfg(
        "na_1d_blackwell_fna_fwd",
        (1024,),
        (256,),
        "blackwell-fna",
        blackwell_only=True,
    ),
    _na_cfg(
        "na_1d_blackwell_fna_bwd",
        (1024,),
        (256,),
        "blackwell-fna",
        bwd=True,
        blackwell_only=True,
    ),
    _na_cfg(
        "na_2d_blackwell_fna",
        (64, 64),
        (32, 32),
        "blackwell-fna",
        blackwell_only=True,
    ),
    _na_cfg(
        "na_3d_blackwell_fna",
        (16, 16, 16),
        (8, 8, 8),
        "blackwell-fna",
        blackwell_only=True,
    ),
    _na_cfg(
        "na_1d_blackwell_strided",
        (1024,),
        (256,),
        "blackwell-fna",
        stride=(128,),
        blackwell_only=True,
    ),
]


# -----------------------------------------------------------------------------
# Problem builders
# -----------------------------------------------------------------------------


def _build_old_problem_na(cfg: ParityConfig) -> OldProblem:
    return OldProblem(
        na_dim=len(cfg.input_size),
        batch_size=cfg.batch_size,
        heads=cfg.heads,
        heads_kv=cfg.heads_kv,
        dim=cfg.dim,
        dim_value=cfg.dim,
        input_size=cfg.input_size,
        window_size=cfg.window_size,
        stride=cfg.stride,
        dilation=cfg.dilation,
        is_causal=cfg.is_causal,
        dtype=cfg.dtype,
    )


def _build_old_problem_sdpa(cfg: ParityConfig) -> OldProblem:
    # _profile_fmha_with_torch ignores window/stride/dilation but requires them.
    return OldProblem(
        na_dim=1,
        batch_size=cfg.batch_size,
        heads=cfg.heads,
        heads_kv=cfg.heads_kv,
        dim=cfg.dim,
        dim_value=cfg.dim,
        input_size=(cfg.seqlen,),
        window_size=(cfg.seqlen,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=cfg.dtype,
    )


def _build_new_na(cfg: ParityConfig) -> NAProblem:
    return NAProblem(
        batch_size=cfg.batch_size,
        heads=cfg.heads,
        heads_kv=cfg.heads_kv,
        dim=cfg.dim,
        dim_value=cfg.dim,
        input_size=cfg.input_size,
        window_size=cfg.window_size,
        stride=cfg.stride,
        dilation=cfg.dilation,
        is_causal=cfg.is_causal,
        dtype=cfg.dtype,
    )


def _build_new_attn(cfg: ParityConfig) -> AttentionProblem:
    return AttentionProblem(
        batch_size=cfg.batch_size,
        heads=cfg.heads,
        heads_kv=cfg.heads_kv,
        dim=cfg.dim,
        dim_value=cfg.dim,
        seqlen_q=cfg.seqlen,
        seqlen_kv=cfg.seqlen,
        dtype=cfg.dtype,
        is_causal=False,
    )


# -----------------------------------------------------------------------------
# Capture helpers
# -----------------------------------------------------------------------------

_PROFILER_ACTIVITY = ProfilerActivity.CUDA if HAS_CUDA else ProfilerActivity.CPU


def _capture_new_trace(
    pool: TensorPool, run_fn: Callable, warmup_steps: int = WARMUP_STEPS
) -> torch_profile:
    """Mirror of profile_op's capture pipeline (without aggregation)."""
    for _ in range(warmup_steps):
        tensors = pool.get()
        if HAS_CUDA:
            torch.cuda.synchronize()
        with torch_profile(activities=[_PROFILER_ACTIVITY]):
            run_fn(tensors)
            if HAS_CUDA:
                torch.cuda.synchronize()

    tensors = pool.get()
    if HAS_CUDA:
        torch.cuda.synchronize()
    with torch_profile(activities=[_PROFILER_ACTIVITY]) as prof:
        run_fn(tensors)
        if HAS_CUDA:
            torch.cuda.synchronize()
    return prof


def _capture_old_trace(cfg: ParityConfig) -> torch_profile:
    if cfg.op == "sdpa":
        return _profile_fmha_with_torch(
            problem=_build_old_problem_sdpa(cfg),
            warmup_steps=WARMUP_STEPS,
            backend=cfg.backend,
            disable_backward=not cfg.bwd,
        )

    if cfg.op == "attn":
        # Old profiler only exposes NA entrypoint; self-attention falls through
        # to FMHA when window_size == input_size.
        problem = OldProblem(
            na_dim=1,
            batch_size=cfg.batch_size,
            heads=cfg.heads,
            heads_kv=cfg.heads_kv,
            dim=cfg.dim,
            dim_value=cfg.dim,
            input_size=(cfg.seqlen,),
            window_size=(cfg.seqlen,),
            stride=(1,),
            dilation=(1,),
            is_causal=(False,),
            dtype=cfg.dtype,
        )
        return _profile_na_with_torch(
            problem=problem,
            warmup_steps=WARMUP_STEPS,
            fmha_backend=cfg.backend,
            disable_backward=not cfg.bwd,
        )

    # NA
    return _profile_na_with_torch(
        problem=_build_old_problem_na(cfg),
        warmup_steps=WARMUP_STEPS,
        backend=cfg.backend,
        disable_backward=not cfg.bwd,
    )


def _build_new_run_fn(cfg: ParityConfig) -> Tuple[TensorPool, Callable]:
    torch.set_grad_enabled(cfg.bwd)

    if cfg.op == "sdpa":
        problem = _build_new_attn(cfg)
        shapes = problem.get_tensor_shapes(heads_last=False)
        pool = TensorPool(
            shapes=shapes,
            dtype=problem.dtype,
            device=torch.device("cuda"),
            init_mode=InitMode.RANDN,
            memory_limit_gb=2.0,
            seed=42,
            requires_grad=cfg.bwd,
        )
        run_fn = partial(
            run_sdpa,
            backend=cfg.backend,
            is_causal=False,
            disable_backward=not cfg.bwd,
        )
        return pool, run_fn

    if cfg.op == "attn":
        problem = _build_new_attn(cfg)
        shapes = problem.get_tensor_shapes(heads_last=True)
        pool = TensorPool(
            shapes=shapes,
            dtype=problem.dtype,
            device=torch.device("cuda"),
            init_mode=InitMode.RANDN,
            memory_limit_gb=2.0,
            seed=42,
            requires_grad=cfg.bwd,
        )
        run_fn = partial(
            run_attn,
            problem=problem,
            backend=cfg.backend,
            disable_backward=not cfg.bwd,
        )
        return pool, run_fn

    # NA
    problem = _build_new_na(cfg)
    shapes = problem.get_tensor_shapes()
    pool = TensorPool(
        shapes=shapes,
        dtype=problem.dtype,
        device=torch.device("cuda"),
        init_mode=InitMode.RANDN,
        memory_limit_gb=2.0,
        seed=42,
        requires_grad=cfg.bwd,
    )
    run_fn = partial(
        run_na,
        problem=problem,
        backend=cfg.backend,
        disable_backward=not cfg.bwd,
    )
    return pool, run_fn


# -----------------------------------------------------------------------------
# Trace extraction
# -----------------------------------------------------------------------------


@dataclass
class RawTrace:
    # symbol -> (num_calls, total_time_us)
    symbols: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    @property
    def total_us(self) -> float:
        return sum(t for _, t in self.symbols.values())


def _extract_raw(prof: torch_profile) -> RawTrace:
    """Extract raw {symbol: (num_calls, time_us)} from a torch profiler trace.

    Skips Memcpy/Memset/CPU-only events. Matches the event-filtering logic in
    both nattenprof.trace.convert_trace_to_results and
    natten.profiling_utils.formatting.convert_to_natten_profiler_ops so the
    comparison is apples-to-apples.
    """
    raw = RawTrace()
    for evt in prof.events():
        if evt.key.startswith("Memcpy") or evt.key.startswith("Memset"):
            continue
        time_us = evt.device_time_total if HAS_CUDA else evt.cpu_time_total
        if time_us <= 0:
            continue
        sym = evt.key
        if sym in raw.symbols:
            c, t = raw.symbols[sym]
            raw.symbols[sym] = (c + 1, t + time_us)
        else:
            raw.symbols[sym] = (1, time_us)
    return raw


# -----------------------------------------------------------------------------
# Canonical normalization
# -----------------------------------------------------------------------------

# Old LibNattenOp.name  ->  canonical tag (chosen to match a new KernelType
# grouping after TokPerm/TokUnperm merge). Used to bucket symbols into the
# same classes on both sides.
OLD_OP_TO_CANONICAL: Dict[LibNattenOp, str] = {
    LibNattenOp.FnaForward: "AttentionForward",
    LibNattenOp.FnaBackward: "AttentionBackward",
    LibNattenOp.FmhaForward: "AttentionForward",
    LibNattenOp.FmhaBackward: "AttentionBackward",
    LibNattenOp.Reduction: "SumOdO",
    LibNattenOp.Elementwise: "Convert",
    LibNattenOp.TokenPermute: "TokenPermute",
}

# New KernelType -> canonical tag. TokPerm + TokUnperm merge into TokenPermute
# because the old profiler does not distinguish them.
NEW_KT_TO_CANONICAL: Dict[KernelType, str] = {
    KernelType.AttentionForward: "AttentionForward",
    KernelType.AttentionBackward: "AttentionBackward",
    KernelType.SumOdO: "SumOdO",
    KernelType.Convert: "Convert",
    KernelType.TokPerm: "TokenPermute",
    KernelType.TokUnperm: "TokenPermute",
    # Init / Misc have no old counterpart — intentionally omitted.
}


def _canonical_counts_from_old(prof: torch_profile) -> Dict[str, int]:
    """Aggregate old profiler's Result list into canonical {tag: num_calls}."""
    counts: Dict[str, int] = {}
    for r in convert_to_natten_profiler_ops(prof) or []:
        if r.op not in OLD_OP_TO_CANONICAL:
            continue
        tag = OLD_OP_TO_CANONICAL[r.op]
        counts[tag] = counts.get(tag, 0) + r.num_calls
    return counts


def _canonical_counts_from_new_raw(raw: RawTrace) -> Dict[str, int]:
    """Use the OLD symbol classifier on raw symbols and map to canonical tags.

    This gives us a classification that we can compare directly to the old
    aggregator's output. We intentionally do NOT use the new classifier here:
    parity is measured against the old profiler's view of the world.
    """
    counts: Dict[str, int] = {}
    for sym, (num_calls, _) in raw.symbols.items():
        op = get_natten_op(sym)
        if op is None or op not in OLD_OP_TO_CANONICAL:
            continue
        tag = OLD_OP_TO_CANONICAL[op]
        counts[tag] = counts.get(tag, 0) + num_calls
    return counts


# -----------------------------------------------------------------------------
# Assertion helpers
# -----------------------------------------------------------------------------


def _format_ratio(a: float, b: float) -> str:
    if min(a, b) <= 0:
        return f"{a:.2f}us vs {b:.2f}us (zero side)"
    return f"{a:.2f}us vs {b:.2f}us (ratio {max(a, b) / min(a, b):.3f})"


def _assert_time_within_tolerance(
    old_us: float, new_us: float, label: str, tolerance: float
):
    """Pass if absolute diff OR relative ratio is within bounds.

    The absolute floor (TIME_NOISE_FLOOR_US) protects against CUDA timer
    quantization biting sub-10us kernels. The ratio bound catches real
    drift on everything larger.
    """
    if old_us < TIME_IGNORE_US and new_us < TIME_IGNORE_US:
        return
    if min(old_us, new_us) <= 0:
        raise AssertionError(
            f"[time/{label}] one side reported non-positive time: "
            f"old={old_us:.3f}us, new={new_us:.3f}us"
        )
    diff = abs(old_us - new_us)
    if diff <= TIME_NOISE_FLOOR_US:
        return
    ratio = max(old_us, new_us) / min(old_us, new_us)
    assert ratio <= tolerance, (
        f"[time/{label}] exceeds ±{(tolerance - 1) * 100:.0f}% and absolute "
        f"diff {diff:.2f}us > {TIME_NOISE_FLOOR_US}us floor: "
        f"old={old_us:.3f}us, new={new_us:.3f}us (ratio={ratio:.3f})"
    )


# -----------------------------------------------------------------------------
# The test
# -----------------------------------------------------------------------------


def _mark(cfg: ParityConfig) -> List[Any]:
    marks = []
    if cfg.hopper_only:
        marks.append(skip_no_hopper)
    elif cfg.blackwell_only:
        marks.append(skip_no_blackwell)
    else:
        marks.append(skip_no_cuda)
    return marks


@pytest.mark.parametrize(
    "cfg",
    [pytest.param(c, id=c.name, marks=_mark(c)) for c in CONFIGS],
)
def test_parity(cfg: ParityConfig):
    from natten import set_memory_usage_preference, use_kv_parallelism_in_fused_na

    use_kv_parallelism_in_fused_na(True)
    set_memory_usage_preference("unrestricted")

    # --- Capture ---
    old_prof = _capture_old_trace(cfg)
    pool, run_fn = _build_new_run_fn(cfg)
    new_prof = _capture_new_trace(pool, run_fn)

    old_raw = _extract_raw(old_prof)
    new_raw = _extract_raw(new_prof)

    # --- 1) Raw symbol sets must be IDENTICAL ---
    old_syms = set(old_raw.symbols)
    new_syms = set(new_raw.symbols)
    missing_in_new = old_syms - new_syms
    missing_in_old = new_syms - old_syms
    assert not missing_in_new and not missing_in_old, (
        f"[{cfg.name}] raw symbol set mismatch.\n"
        f"  present in OLD but not NEW ({len(missing_in_new)}): "
        f"{sorted(list(missing_in_new))[:5]}\n"
        f"  present in NEW but not OLD ({len(missing_in_old)}): "
        f"{sorted(list(missing_in_old))[:5]}"
    )

    # --- 2) num_calls per symbol must be IDENTICAL ---
    mismatches = []
    for sym in old_syms:
        old_calls, _ = old_raw.symbols[sym]
        new_calls, _ = new_raw.symbols[sym]
        if old_calls != new_calls:
            mismatches.append((sym, old_calls, new_calls))
    assert not mismatches, (
        f"[{cfg.name}] num_calls mismatch on {len(mismatches)} symbol(s):\n"
        + "\n".join(f"  {o} vs {n} for {s[:100]}" for s, o, n in mismatches[:5])
    )

    # --- 3) per-symbol time must be within TIME_TOLERANCE_RATIO ---
    for sym in old_syms:
        _, old_t = old_raw.symbols[sym]
        _, new_t = new_raw.symbols[sym]
        _assert_time_within_tolerance(
            old_t, new_t, label=f"{cfg.name}/{sym[:40]}", tolerance=TIME_TOLERANCE_RATIO
        )

    # --- 4) total time must be within TIME_TOLERANCE_RATIO ---
    _assert_time_within_tolerance(
        old_raw.total_us,
        new_raw.total_us,
        label=f"{cfg.name}/total",
        tolerance=TIME_TOLERANCE_RATIO,
    )

    # --- 5) canonical op classification must be IDENTICAL ---
    # Use the OLD classifier on BOTH sides (raw symbols -> canonical tag) and
    # additionally compare with old profiler's own aggregation. This ensures:
    #   - the old classifier sees the same kernels on both sides
    #   - the old profiler's own aggregation matches our reconstruction
    old_canonical = _canonical_counts_from_old(old_prof)
    new_canonical = _canonical_counts_from_new_raw(new_raw)

    assert old_canonical == new_canonical, (
        f"[{cfg.name}] canonical op count mismatch.\n"
        f"  OLD: {old_canonical}\n"
        f"  NEW: {new_canonical}"
    )
