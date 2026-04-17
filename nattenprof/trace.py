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

"""Trace event filtering and symbol-to-KernelResult mapping."""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from natten.utils import log
from torch.profiler import profile as torch_profile

from nattenprof.output import KernelResult

logger = log.get_logger(__name__)


class KernelType(Enum):
    AttentionForward = "attention_forward"
    AttentionBackward = "attention_backward"
    SumOdO = "sum_odo"
    Convert = "convert"
    TokPerm = "tok_perm"
    TokUnperm = "tok_unperm"
    Init = "init"
    Misc = "misc"


class Category(Enum):
    AttnFwd = "attn_fwd"
    AttnBwd = "attn_bwd"
    MemoryOps = "memory_ops"
    Misc = "misc"


KERNEL_TYPE_TO_CATEGORY = {
    KernelType.AttentionForward: Category.AttnFwd,
    KernelType.AttentionBackward: Category.AttnBwd,
    KernelType.SumOdO: Category.AttnBwd,
    KernelType.Convert: Category.AttnBwd,
    KernelType.TokPerm: Category.MemoryOps,
    KernelType.TokUnperm: Category.MemoryOps,
    KernelType.Init: Category.Misc,
    KernelType.Misc: Category.Misc,
}

# Display priority (lower = higher in table).
KERNEL_TYPE_PRIORITY = {
    KernelType.AttentionForward: 0,
    KernelType.AttentionBackward: 1,
    KernelType.SumOdO: 2,
    KernelType.Convert: 3,
    KernelType.TokPerm: 4,
    KernelType.TokUnperm: 5,
    KernelType.Init: 6,
    KernelType.Misc: 99,
}

CATEGORY_PRIORITY = {
    Category.AttnFwd: 0,
    Category.AttnBwd: 1,
    Category.MemoryOps: 2,
    Category.Misc: 99,
}


# ---- Symbol registry ----
#
# Each entry: (pattern, KernelType)
#
# Pattern is either:
#   - str: simple substring match
#   - tuple of (str | tuple[str, ...]): hierarchical AND/OR match.
#     Top-level tuple elements are ANDed (all must match the symbol).
#     Nested tuple elements are ORed (any one must match).
#     Max nesting depth: 2 (no tuple inside tuple inside tuple).
#
# Checked in order; first match wins.

SymbolPattern = Union[str, Tuple[Union[str, Tuple[str, ...]], ...]]

SYMBOL_REGISTRY: List[Tuple[SymbolPattern, KernelType]] = [
    # ---- NATTEN FNA ----
    (("natten::cuda::fna::", "BackwardKernel"), KernelType.AttentionBackward),
    ("natten::cuda::fna::", KernelType.AttentionForward),
    (("cutlass::fna::", "Bwd"), KernelType.AttentionBackward),
    ("cutlass::fna::", KernelType.AttentionForward),
    #
    # ---- NATTEN FMHA ----
    # Backward auxiliaries must come before generic FMHA backward (more specific first).
    (("natten::cuda::fmha::", "BackwardKernel"), KernelType.AttentionBackward),
    ("natten::cuda::fmha::", KernelType.AttentionForward),
    ("natten::cuda::reduction::kernel::ComputeDelta", KernelType.SumOdO),
    (("cutlass::fmha::kernel::FmhaKernel", "SumOdO"), KernelType.SumOdO),
    (("cutlass::fmha::kernel::FmhaKernel", "Convert"), KernelType.Convert),
    (("cutlass::fmha::", ("Bwd", "Backward")), KernelType.AttentionBackward),
    ("cutlass::fmha::collective::FmhaMainloop", KernelType.AttentionForward),
    # Blackwell FMHA: kernel-level symbol, no collective:: wrapper.
    ("cutlass::fmha::kernel::Sm100FmhaFwd", KernelType.AttentionForward),
    #
    # ---- cuDNN ----
    (("cudnn", "compute_dot_do_o"), KernelType.SumOdO),
    (("cudnn", "convert_dq"), KernelType.Convert),
    (("cudnn", ("flash", "sdpa", "fmha"), "bprop"), KernelType.AttentionBackward),
    (("cudnn", ("flash", "sdpa", "fmha"), "fprop"), KernelType.AttentionForward),
    #
    # ---- Flash Attention v2 (via torch SDPA) ----
    (("pytorch_flash::", "dot_do_o"), KernelType.SumOdO),
    (("pytorch_flash::", "convert_dq"), KernelType.Convert),
    (("pytorch_flash::", "bwd"), KernelType.AttentionBackward),
    (("pytorch_flash::", "fwd"), KernelType.AttentionForward),
    #
    # ---- Token permute ----
    ("natten::tokperm::kernel", None),  # special: perm/unperm inferred below
    #
    # ---- Init (zero-fill / workspace init) ----
    ("FillFunctor", KernelType.Init),
    ("fill_kernel", KernelType.Init),
]


# ---- Namespace detection (symbol-based) ----
# Check in listed order, first match wins.
# "natten (only if not cutlass)" and "flash (only if not cudnn)" qualifiers are
# automatically satisfied by the listed ordering (CUTLASS and cuDNN checked first).

_PYTORCH_MARKERS = ("aten", "at::", "c10::", "torch", "pytorch")

# ---- Arch detection ----

ARCH_LOOKUP = {
    "Sm80": ["sm80"],
    "Sm86": ["sm86"],
    "Sm8X": ["ampere"],
    "Sm89": ["sm89"],
    "Sm90": ["sm90", "hopper"],
    "Sm100": ["sm100", "blackwell"],
    "Sm120": ["sm120"],
}


def _match_pattern(symbol: str, pattern: SymbolPattern) -> bool:
    """Match a symbol against a pattern (str or hierarchical AND/OR tuple)."""
    if isinstance(pattern, str):
        return pattern in symbol

    # Tuple: top-level = AND, nested tuple = OR
    assert isinstance(pattern, tuple), f"Invalid pattern type: {type(pattern)}"
    for element in pattern:
        if isinstance(element, str):
            if element not in symbol:
                return False
        elif isinstance(element, tuple):
            # OR: any element must match
            if not any(sub in symbol for sub in element):
                return False
        else:
            raise ValueError(
                f"Invalid pattern element type: {type(element)}. "
                "Expected str or tuple of str."
            )
    return True


def _match_kernel_type(symbol: str) -> Optional[KernelType]:
    """Match symbol to KernelType. Returns None if no pattern matches."""
    for pattern, ktype in SYMBOL_REGISTRY:
        if _match_pattern(symbol, pattern):
            if ktype is not None:
                return ktype

            # Special case: tokperm — infer perm/unperm from is_inverse template param
            if "natten::tokperm::kernel" in symbol:
                return _classify_tokperm(symbol)

    return None


def _classify_tokperm(symbol: str) -> KernelType:
    """Classify token permute kernel as TokPerm or TokUnperm.

    The symbol contains TokenPermuteKernel<..., is_inverse, alignment>.
    is_inverse: false=perm, true=unperm (second-to-last template param).
    """
    lower = symbol.lower()
    if ", true," in lower:
        last_true = lower.rfind(", true,")
        last_angle = lower.rfind(">")
        if last_true > 0 and (last_angle - last_true) < 20:
            return KernelType.TokUnperm
    return KernelType.TokPerm


def _get_namespace(symbol: str) -> str:
    """Classify kernel namespace from symbol.

    Checks in order:
      cuDNN   -- symbol contains "cudnn"
      PyTorch -- symbol contains any of: aten, at::, c10::, torch, pytorch
      CUTLASS -- symbol contains "cutlass"
      NATTEN  -- symbol contains "natten" (and not cutlass, by ordering)
      flash   -- symbol contains "flash" (and not cudnn, by ordering)
    """
    lower = symbol.lower()
    if "cudnn" in lower:
        return "cuDNN"
    if any(m in lower for m in _PYTORCH_MARKERS):
        return "PyTorch"
    if "cutlass" in lower:
        return "CUTLASS"
    if "natten" in lower:
        return "NATTEN"
    if "flash" in lower:
        return "flash"
    return "-"


def _get_arch(symbol: str) -> str:
    lower = symbol.lower()
    for name, tags in ARCH_LOOKUP.items():
        for tag in tags:
            if tag in lower:
                return name
    return "-"


def _get_op_name(symbol: str, kernel_type: KernelType) -> str:
    """Derive operation name. Distinguishes FNA vs FMHA for attention kernels."""
    if kernel_type == KernelType.AttentionForward:
        if _is_fna_symbol(symbol):
            return "FNAForward"
        return "FMHAForward"

    if kernel_type == KernelType.AttentionBackward:
        if _is_fna_symbol(symbol):
            return "FNABackward"
        return "FMHABackward"

    if kernel_type != KernelType.Misc:
        return kernel_type.name

    # Fallback for Misc: strip wrappers
    name = symbol.strip()
    for prefix in ("void (anonymous namespace)::", "void "):
        if name.startswith(prefix):
            name = name[len(prefix) :]

    name = name.split("<")[0].split(">")[0]
    name = name.replace("(anonymous namespace)::", "")
    name = name.replace("at::native::", "")

    if name.startswith("cudnn_generated_fort_native_"):
        name = name.replace("cudnn_generated_fort_native_", "")

    if name.startswith("triton_"):
        name = name.replace("triton_", "")

    parts = name.split("::")
    if len(parts) > 1:
        return ".".join(parts)

    return name


def _is_fna_symbol(symbol: str) -> bool:
    """Check if symbol is from an FNA kernel (vs FMHA).

    Hopper/Blackwell FNA kernels use the FMHA kernel wrapper
    (cutlass::fmha::kernel::FmhaKernelTmaWarpSpecialized) but contain
    FNA-specific collectives (cutlass::fna::collective) inside the template.
    """
    return "cutlass::fna::" in symbol or "natten::cuda::fna::" in symbol


def _guess_kernel_type(symbol: str) -> KernelType:
    """Heuristic for unrecognized symbols. Always returns Misc."""
    logger.debug(f"Unrecognized kernel symbol classified as Misc: {symbol[:120]}")
    return KernelType.Misc


def convert_trace_to_results(
    profiler: torch_profile,
) -> Optional[List[KernelResult]]:
    """Convert torch profiler trace events into KernelResult list."""
    events = profiler.events()

    aggregated: Dict[str, KernelResult] = {}

    for evt in events:
        if evt.key.startswith("Memcpy") or evt.key.startswith("Memset"):
            continue

        time_total = (
            evt.device_time_total if torch.cuda.is_available() else evt.cpu_time_total
        )

        if time_total <= 0:
            continue

        symbol = evt.key
        kernel_type = _match_kernel_type(symbol)
        if kernel_type is None:
            kernel_type = _guess_kernel_type(symbol)

        namespace = _get_namespace(symbol)
        arch = _get_arch(symbol)
        op_name = _get_op_name(symbol, kernel_type)

        result = KernelResult(
            kernel_type=kernel_type,
            namespace=namespace,
            arch=arch,
            op_name=op_name,
            symbol=symbol,
            num_calls=1,
            time_us=time_total,
        )

        # Dedup key: same display identity -> aggregate
        key = f"{namespace}_{kernel_type.value}_{arch}_{op_name}"
        if key in aggregated:
            existing = aggregated[key]
            aggregated[key] = KernelResult(
                kernel_type=existing.kernel_type,
                namespace=existing.namespace,
                arch=existing.arch,
                op_name=existing.op_name,
                symbol=existing.symbol,
                num_calls=existing.num_calls + 1,
                time_us=existing.time_us + time_total,
            )
        else:
            aggregated[key] = result

    if not aggregated:
        return None

    results = sorted(
        aggregated.values(),
        key=lambda r: KERNEL_TYPE_PRIORITY.get(r.kernel_type, 99),
    )
    return results
