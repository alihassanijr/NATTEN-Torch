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

"""Core profiler tests: call profile_op() directly, verify expected kernel types,
packages, frameworks, and op names."""

from functools import partial
from typing import List

import pytest
import torch
from natten.utils import log
from natten.utils.device import get_device_cc

from nattenprof.engine import profile_op
from nattenprof.ops import run_attn, run_na, run_sdpa
from nattenprof.output import KernelResult
from nattenprof.problem import AttentionProblem, NAProblem
from nattenprof.tensors import InitMode, TensorPool
from nattenprof.trace import KernelType

logger = log.get_logger("nattenprof_tests")

HAS_CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _get_cc():
    if not HAS_CUDA:
        return 0
    return get_device_cc()


skip_no_hopper = pytest.mark.skipif(
    not HAS_CUDA or _get_cc() != 90,
    reason="Hopper kernels require SM90",
)

skip_no_blackwell = pytest.mark.skipif(
    not HAS_CUDA or _get_cc() not in (100, 103),
    reason="Blackwell kernels require SM100 or SM103",
)

skip_no_libnatten = pytest.mark.skipif(
    not HAS_CUDA,
    reason="Requires libnatten + CUDA",
)


def _setup():
    from natten import set_memory_usage_preference, use_kv_parallelism_in_fused_na

    use_kv_parallelism_in_fused_na(True)
    set_memory_usage_preference("unrestricted")


def _make_pool(problem, heads_last=True, requires_grad=False):
    shapes = (
        problem.get_tensor_shapes(heads_last=heads_last)
        if isinstance(problem, AttentionProblem)
        else problem.get_tensor_shapes()
    )
    return TensorPool(
        shapes=shapes,
        dtype=problem.dtype,
        device=torch.device("cuda"),
        init_mode=InitMode.RANDN,
        memory_limit_gb=2.0,
        seed=42,
        requires_grad=requires_grad,
    )


def _count(results: List[KernelResult], kt: KernelType) -> int:
    return sum(r.num_calls for r in results if r.kernel_type == kt)


def _has(results: List[KernelResult], kt: KernelType) -> bool:
    return _count(results, kt) > 0


def _get_by_type(results: List[KernelResult], kt: KernelType) -> List[KernelResult]:
    return [r for r in results if r.kernel_type == kt]


def _assert_namespace(results: List[KernelResult], kt: KernelType, expected: str):
    for r in _get_by_type(results, kt):
        assert r.namespace == expected, (
            f"Expected namespace={expected} for {kt.name}, "
            f"got {r.namespace} (symbol: {r.symbol[:80]})"
        )


def _assert_op_name(results: List[KernelResult], kt: KernelType, expected: str):
    for r in _get_by_type(results, kt):
        assert r.op_name == expected, (
            f"Expected op_name={expected} for {kt.name}, "
            f"got {r.op_name} (symbol: {r.symbol[:80]})"
        )


def _assert_symbol_contains(
    results: List[KernelResult], kt: KernelType, expected_substring: str
):
    """Verify raw kernel symbol contains expected substring.

    Guards against classification bugs (e.g. false-positive that assigns the right
    namespace/op_name to a kernel whose actual symbol doesn't match).
    """
    matches = _get_by_type(results, kt)
    assert len(matches) > 0, f"No kernels of type {kt.name} found"
    for r in matches:
        assert expected_substring in r.symbol, (
            f"Expected substring {expected_substring!r} in symbol for {kt.name}, "
            f"got symbol: {r.symbol}"
        )


def _log_results(results: List[KernelResult]):
    logger.debug(
        "Kernels: "
        + str(
            [(r.kernel_type.name, r.op_name, r.namespace, r.num_calls) for r in results]
        )
    )


# ---- NA: hopper-fna fwd ----


@skip_no_hopper
def test_na_1d_hopper_fna_fwd():
    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(1024,),
        window_size=(256,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, requires_grad=False)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="hopper-fna",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    # Kernel counts
    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.TokPerm) == 3  # q, k, v
    assert _count(results, KernelType.TokUnperm) == 2  # lse, out
    assert not _has(results, KernelType.AttentionBackward)
    assert not _has(results, KernelType.SumOdO)
    assert not _has(results, KernelType.Convert)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_namespace(results, KernelType.TokPerm, "CUTLASS")

    _assert_op_name(results, KernelType.AttentionForward, "FNAForward")
    _assert_op_name(results, KernelType.TokPerm, "TokPerm")
    _assert_op_name(results, KernelType.TokUnperm, "TokUnperm")

    # Raw symbols must match known hopper-fna patterns
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fna::collective::FnaMainloop"
    )
    _assert_symbol_contains(
        results, KernelType.TokPerm, "natten::tokperm::kernel::TokenPermuteKernel"
    )
    _assert_symbol_contains(
        results, KernelType.TokUnperm, "natten::tokperm::kernel::TokenPermuteKernel"
    )


# ---- NA: hopper-fna fwd+bwd ----


@skip_no_hopper
def test_na_1d_hopper_fna_bwd():
    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(1024,),
        window_size=(256,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, requires_grad=True)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="hopper-fna",
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    # Kernel counts
    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert _count(results, KernelType.SumOdO) == 1
    assert _count(results, KernelType.Convert) == 1
    # fwd: 3 perm + 2 unperm = 5, bwd: 6 perm + 3 unperm = 9, total = 14
    total_mem_ops = _count(results, KernelType.TokPerm) + _count(
        results, KernelType.TokUnperm
    )
    assert total_mem_ops == 14
    assert _has(results, KernelType.Init)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_namespace(results, KernelType.AttentionBackward, "CUTLASS")
    _assert_namespace(results, KernelType.SumOdO, "CUTLASS")
    _assert_namespace(results, KernelType.Convert, "CUTLASS")
    _assert_namespace(results, KernelType.Init, "PyTorch")

    _assert_op_name(results, KernelType.AttentionForward, "FNAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FNABackward")

    # Raw symbols must match known hopper-fna + bwd-aux patterns
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fna::collective::FnaMainloop"
    )
    _assert_symbol_contains(
        results,
        KernelType.AttentionBackward,
        "cutlass::fna::collective::FnaBwdMainloop",
    )
    _assert_symbol_contains(
        results, KernelType.SumOdO, "cutlass::fmha::kernel::FmhaKernelBwdSumOdO"
    )
    _assert_symbol_contains(
        results, KernelType.Convert, "cutlass::fmha::kernel::FmhaKernelBwdConvert"
    )
    _assert_symbol_contains(results, KernelType.Init, "FillFunctor")


# ---- NA: cutlass-fna fwd ----


@skip_no_libnatten
def test_na_1d_cutlass_fna_fwd():
    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(1024,),
        window_size=(256,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, requires_grad=False)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="cutlass-fna",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert not _has(results, KernelType.TokPerm)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_op_name(results, KernelType.AttentionForward, "FNAForward")

    # Raw symbol must match cutlass-fna pattern (natten namespace, Sm80 kernel)
    _assert_symbol_contains(
        results,
        KernelType.AttentionForward,
        "natten::cuda::fna::FusedNeighborhoodAttentionKernel",
    )


# ---- NA: cutlass-fna fwd+bwd ----


@skip_no_libnatten
def test_na_1d_cutlass_fna_bwd():
    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(1024,),
        window_size=(256,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, requires_grad=True)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="cutlass-fna",
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert _count(results, KernelType.SumOdO) == 1
    assert not _has(results, KernelType.Convert)  # cutlass-fna fuses convert
    assert not _has(results, KernelType.TokPerm)

    _assert_op_name(results, KernelType.AttentionForward, "FNAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FNABackward")

    # Raw symbols must match known cutlass-fna patterns
    _assert_symbol_contains(
        results,
        KernelType.AttentionForward,
        "natten::cuda::fna::FusedNeighborhoodAttentionKernel",
    )
    _assert_symbol_contains(
        results,
        KernelType.AttentionBackward,
        "natten::cuda::fna::FusedNeighborhoodAttentionBackwardKernel",
    )
    # cutlass-fna reuses the cutlass FMHA bwd SumOdO kernel
    _assert_symbol_contains(
        results, KernelType.SumOdO, "cutlass::fmha::kernel::FmhaKernelBwdSumOdO"
    )


# ---- Attn: hopper-fmha fwd ----


@skip_no_hopper
def test_attn_hopper_fmha_fwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, heads_last=True, requires_grad=False)
    run_fn = partial(
        run_attn,
        problem=problem,
        backend="hopper-fmha",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert not _has(results, KernelType.AttentionBackward)
    assert not _has(results, KernelType.TokPerm)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")

    # Raw symbol must match hopper-fmha pattern
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fmha::collective::FmhaMainloop"
    )


# ---- Attn: hopper-fmha fwd+bwd ----


@skip_no_hopper
def test_attn_hopper_fmha_bwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, heads_last=True, requires_grad=True)
    run_fn = partial(
        run_attn,
        problem=problem,
        backend="hopper-fmha",
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert _count(results, KernelType.SumOdO) == 1
    assert _count(results, KernelType.Convert) == 1
    assert not _has(results, KernelType.TokPerm)

    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FMHABackward")

    # Raw symbols must match hopper-fmha + bwd-aux patterns
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fmha::collective::FmhaMainloop"
    )
    _assert_symbol_contains(
        results,
        KernelType.AttentionBackward,
        "cutlass::fmha::collective::FmhaBwdMainloop",
    )
    _assert_symbol_contains(
        results, KernelType.SumOdO, "cutlass::fmha::kernel::FmhaKernelBwdSumOdO"
    )
    _assert_symbol_contains(
        results, KernelType.Convert, "cutlass::fmha::kernel::FmhaKernelBwdConvert"
    )


# ---- Attn: cutlass-fmha fwd ----


@skip_no_libnatten
def test_attn_cutlass_fmha_fwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, heads_last=True, requires_grad=False)
    run_fn = partial(
        run_attn,
        problem=problem,
        backend="cutlass-fmha",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert not _has(results, KernelType.AttentionBackward)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")

    # Raw symbol must match cutlass-fmha pattern
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "natten::cuda::fmha::AttentionKernel"
    )


# ---- Attn: cutlass-fmha fwd+bwd ----


@skip_no_libnatten
def test_attn_cutlass_fmha_bwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, heads_last=True, requires_grad=True)
    run_fn = partial(
        run_attn,
        problem=problem,
        backend="cutlass-fmha",
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert _count(results, KernelType.SumOdO) == 1
    assert not _has(results, KernelType.Convert)  # cutlass-fmha fuses convert
    assert not _has(results, KernelType.TokPerm)

    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FMHABackward")

    # Raw symbols must match cutlass-fmha patterns
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "natten::cuda::fmha::AttentionKernel"
    )
    _assert_symbol_contains(
        results,
        KernelType.AttentionBackward,
        "natten::cuda::fmha::AttentionBackwardKernel",
    )
    # cutlass-fmha reuses the cutlass FMHA bwd SumOdO kernel
    _assert_symbol_contains(
        results, KernelType.SumOdO, "cutlass::fmha::kernel::FmhaKernelBwdSumOdO"
    )


# ---- SDPA: cudnn fwd ----


@skip_no_cuda
def test_sdpa_cudnn_fwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, heads_last=False, requires_grad=False)
    run_fn = partial(
        run_sdpa,
        backend="cudnn",
        is_causal=False,
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert not _has(results, KernelType.AttentionBackward)

    _assert_namespace(results, KernelType.AttentionForward, "cuDNN")
    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")

    # Raw symbol must match cudnn SDPA fprop pattern
    _assert_symbol_contains(results, KernelType.AttentionForward, "cudnn")
    _assert_symbol_contains(results, KernelType.AttentionForward, "fprop")


# ---- SDPA: cudnn fwd+bwd ----


@skip_no_cuda
def test_sdpa_cudnn_bwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, heads_last=False, requires_grad=True)
    run_fn = partial(
        run_sdpa,
        backend="cudnn",
        is_causal=False,
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert _count(results, KernelType.SumOdO) == 1
    assert _count(results, KernelType.Convert) == 1

    _assert_namespace(results, KernelType.AttentionForward, "cuDNN")
    _assert_namespace(results, KernelType.AttentionBackward, "cuDNN")
    _assert_namespace(results, KernelType.SumOdO, "cuDNN")
    _assert_namespace(results, KernelType.Convert, "cuDNN")
    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FMHABackward")

    # Raw symbols must match cudnn SDPA patterns
    _assert_symbol_contains(results, KernelType.AttentionForward, "cudnn")
    _assert_symbol_contains(results, KernelType.AttentionForward, "fprop")
    _assert_symbol_contains(results, KernelType.AttentionBackward, "cudnn")
    _assert_symbol_contains(results, KernelType.AttentionBackward, "bprop")
    _assert_symbol_contains(results, KernelType.SumOdO, "compute_dot_do_o")
    _assert_symbol_contains(results, KernelType.Convert, "convert_dq")


# ---- SDPA: fav2 fwd ----


@skip_no_cuda
def test_sdpa_fav2_fwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.float16,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, heads_last=False, requires_grad=False)
    run_fn = partial(
        run_sdpa,
        backend="fav2",
        is_causal=False,
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    # fav2 may use splitkv (2 kernels) or single kernel depending on seqlen
    assert _count(results, KernelType.AttentionForward) >= 1

    _assert_namespace(results, KernelType.AttentionForward, "PyTorch")
    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")

    # Raw symbol must match pytorch flash fwd pattern
    _assert_symbol_contains(results, KernelType.AttentionForward, "pytorch_flash::")
    _assert_symbol_contains(results, KernelType.AttentionForward, "fwd")


# ---- NA: flex-fna fwd ----


@skip_no_cuda
def test_na_1d_flex_fna_fwd():
    """Flex backend fwd-only: just verify it runs without crashing.

    Unfused flex (without torch.compile) decomposes into many PyTorch native
    kernels that aren't classifiable as attention. We only check it doesn't
    crash and produces some results.
    """
    try:
        from natten.backends.flex import _FLEX_SUPPORTED

        if not _FLEX_SUPPORTED:
            pytest.skip("Flex backend not supported.")
    except ImportError:
        pytest.skip("Flex backend not available.")

    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(512,),
        window_size=(128,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.float16,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, requires_grad=False)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="flex-fna",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    assert len(results) > 0
    assert not _has(results, KernelType.TokPerm)  # flex doesn't use tokperm


# ---- NA: blackwell-fna fwd ----


@skip_no_blackwell
def test_na_1d_blackwell_fna_fwd():
    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(1024,),
        window_size=(256,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, requires_grad=False)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="blackwell-fna",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _has(results, KernelType.TokPerm)
    assert _has(results, KernelType.TokUnperm)
    assert not _has(results, KernelType.AttentionBackward)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_namespace(results, KernelType.TokPerm, "CUTLASS")

    _assert_op_name(results, KernelType.AttentionForward, "FNAForward")
    _assert_op_name(results, KernelType.TokPerm, "TokPerm")
    _assert_op_name(results, KernelType.TokUnperm, "TokUnperm")

    # Raw symbols must match known blackwell-fna patterns
    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fna::kernel::Sm100FnaFwd"
    )
    _assert_symbol_contains(
        results, KernelType.TokPerm, "natten::tokperm::kernel::TokenPermuteKernel"
    )
    _assert_symbol_contains(
        results, KernelType.TokUnperm, "natten::tokperm::kernel::TokenPermuteKernel"
    )


# ---- NA: blackwell-fna fwd+bwd ----


@skip_no_blackwell
def test_na_1d_blackwell_fna_bwd():
    _setup()
    problem = NAProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        input_size=(1024,),
        window_size=(256,),
        stride=(1,),
        dilation=(1,),
        is_causal=(False,),
        dtype=torch.bfloat16,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, requires_grad=True)
    run_fn = partial(
        run_na,
        problem=problem,
        backend="blackwell-fna",
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert _has(results, KernelType.TokPerm)
    assert _has(results, KernelType.TokUnperm)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_namespace(results, KernelType.AttentionBackward, "CUTLASS")

    _assert_op_name(results, KernelType.AttentionForward, "FNAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FNABackward")

    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fna::kernel::Sm100FnaFwd"
    )
    _assert_symbol_contains(
        results, KernelType.AttentionBackward, "cutlass::fna::kernel::Sm100FnaBwd"
    )


# ---- Attn: blackwell-fmha fwd ----


@skip_no_blackwell
def test_attn_blackwell_fmha_fwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    torch.set_grad_enabled(False)
    pool = _make_pool(problem, heads_last=True, requires_grad=False)
    run_fn = partial(
        run_attn,
        problem=problem,
        backend="blackwell-fmha",
        disable_backward=True,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert not _has(results, KernelType.AttentionBackward)
    assert not _has(results, KernelType.TokPerm)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")

    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fmha::kernel::Sm100FmhaFwd"
    )


# ---- Attn: blackwell-fmha fwd+bwd ----


@skip_no_blackwell
def test_attn_blackwell_fmha_bwd():
    problem = AttentionProblem(
        batch_size=1,
        heads=1,
        heads_kv=1,
        dim=128,
        dim_value=128,
        seqlen_q=1024,
        seqlen_kv=1024,
        dtype=torch.bfloat16,
        is_causal=False,
    )

    torch.set_grad_enabled(True)
    pool = _make_pool(problem, heads_last=True, requires_grad=True)
    run_fn = partial(
        run_attn,
        problem=problem,
        backend="blackwell-fmha",
        disable_backward=False,
    )

    results = profile_op(pool=pool, run_fn=run_fn, warmup_steps=5)
    assert results is not None
    _log_results(results)

    assert _count(results, KernelType.AttentionForward) == 1
    assert _count(results, KernelType.AttentionBackward) == 1
    assert not _has(results, KernelType.TokPerm)

    _assert_namespace(results, KernelType.AttentionForward, "CUTLASS")
    _assert_namespace(results, KernelType.AttentionBackward, "CUTLASS")

    _assert_op_name(results, KernelType.AttentionForward, "FMHAForward")
    _assert_op_name(results, KernelType.AttentionBackward, "FMHABackward")

    _assert_symbol_contains(
        results, KernelType.AttentionForward, "cutlass::fmha::kernel::Sm100FmhaFwd"
    )
    _assert_symbol_contains(
        results, KernelType.AttentionBackward, "cutlass::fmha::kernel::Sm100FmhaBwd"
    )
