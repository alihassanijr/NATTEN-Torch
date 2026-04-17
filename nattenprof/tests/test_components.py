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

"""Component-level tests: symbol classification, tensor pool, problem classes.
These do NOT require GPU (except TensorPool tests)."""

import pytest
import torch

from nattenprof.problem import AttentionProblem, NAProblem
from nattenprof.trace import (
    _get_arch,
    _get_namespace,
    _get_op_name,
    _is_fna_symbol,
    _match_kernel_type,
    _match_pattern,
    Category,
    KERNEL_TYPE_TO_CATEGORY,
    KernelType,
)

HAS_CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# ---- Symbol classification ----


class TestSymbolClassification:
    """Test _match_kernel_type for all known symbol patterns."""

    # NATTEN FNA
    def test_cutlass_fna_fwd(self):
        sym = (
            "natten::cuda::fna::fna1d_64x64x128_sm80_bfloat16_cm_0"
            "(natten::cuda::fna::FusedNeighborhoodAttentionKernel<...>)"
        )
        assert _match_kernel_type(sym) == KernelType.AttentionForward

    def test_cutlass_fna_bwd(self):
        sym = (
            "natten::cuda::fna::fna1d_backward_64x64x128_sm80_bfloat16_cm_0"
            "(natten::cuda::fna::FusedNeighborhoodAttentionBackwardKernel<...>)"
        )
        assert _match_kernel_type(sym) == KernelType.AttentionBackward

    def test_hopper_fna_fwd(self):
        sym = (
            "void cutlass::device_kernel_sm90<cutlass::fmha::kernel::"
            "FmhaKernelTmaWarpSpecialized<..., cutlass::fna::collective::"
            "FnaMainloopTmaWarpSpecializedSm90<...>>>"
        )
        assert _match_kernel_type(sym) == KernelType.AttentionForward

    def test_hopper_fna_bwd(self):
        sym = (
            "void cutlass::device_kernel_sm90<cutlass::fmha::kernel::"
            "FmhaKernelTmaWarpSpecialized<..., cutlass::fna::collective::"
            "FnaBwdMainloopTmaWarpSpecializedSm90<...>>>"
        )
        assert _match_kernel_type(sym) == KernelType.AttentionBackward

    # NATTEN FMHA
    def test_cutlass_fmha_fwd(self):
        sym = "natten::cuda::fmha::AttentionKernel<cutlass::bfloat16_t, ...>"
        assert _match_kernel_type(sym) == KernelType.AttentionForward

    def test_cutlass_fmha_bwd(self):
        sym = "natten::cuda::fmha::AttentionBackwardKernel<cutlass::bfloat16_t, ...>"
        assert _match_kernel_type(sym) == KernelType.AttentionBackward

    def test_hopper_fmha_fwd(self):
        sym = (
            "void cutlass::device_kernel_sm90<cutlass::fmha::kernel::"
            "FmhaKernelTmaWarpSpecialized<..., cutlass::fmha::collective::"
            "FmhaMainloopTmaWarpSpecializedSm90<...>>>"
        )
        assert _match_kernel_type(sym) == KernelType.AttentionForward

    # cuDNN
    def test_cudnn_fprop(self):
        sym = "cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7"
        assert _match_kernel_type(sym) == KernelType.AttentionForward

    def test_cudnn_bprop(self):
        sym = "cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26"
        assert _match_kernel_type(sym) == KernelType.AttentionBackward

    def test_cudnn_sum_odo(self):
        sym = "void cudnn::fusion::compute_dot_do_o_specialized<true, 128>(...)"
        assert _match_kernel_type(sym) == KernelType.SumOdO

    def test_cudnn_convert(self):
        sym = "void cudnn::fusion::convert_dq_to_16bits<true>(...)"
        assert _match_kernel_type(sym) == KernelType.Convert

    # Flash Attention v2
    def test_fav2_fwd(self):
        sym = "void pytorch_flash::flash_fwd_splitkv_kernel<...>"
        assert _match_kernel_type(sym) == KernelType.AttentionForward

    def test_fav2_bwd(self):
        sym = "void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<...>"
        assert _match_kernel_type(sym) == KernelType.AttentionBackward

    def test_fav2_sum_odo(self):
        sym = "void pytorch_flash::flash_bwd_dot_do_o_kernel<true, ...>"
        assert _match_kernel_type(sym) == KernelType.SumOdO

    def test_fav2_convert(self):
        sym = "void pytorch_flash::flash_bwd_convert_dq_kernel<...>"
        assert _match_kernel_type(sym) == KernelType.Convert

    # Backward auxiliaries (NATTEN)
    def test_natten_sum_odo(self):
        sym = (
            "void cutlass::device_kernel_sm90<cutlass::fmha::kernel::"
            "FmhaKernelBwdSumOdO<...>>"
        )
        assert _match_kernel_type(sym) == KernelType.SumOdO

    def test_natten_convert(self):
        sym = (
            "void cutlass::device_kernel_sm90<cutlass::fmha::kernel::"
            "FmhaKernelBwdConvert<...>>"
        )
        assert _match_kernel_type(sym) == KernelType.Convert

    # Token permute
    def test_tokperm(self):
        sym = (
            "void cutlass::device_kernel<natten::tokperm::kernel::"
            "TokenPermuteKernel<..., false, 8>>"
        )
        assert _match_kernel_type(sym) == KernelType.TokPerm

    # Init
    def test_fill_functor(self):
        sym = (
            "void at::native::vectorized_elementwise_kernel<4, "
            "at::native::FillFunctor<float>, ...>"
        )
        assert _match_kernel_type(sym) == KernelType.Init

    # Misc (unrecognized)
    def test_unrecognized(self):
        sym = "void some_random_kernel<float>(int)"
        assert _match_kernel_type(sym) is None


# ---- FNA vs FMHA distinction ----


class TestFnaDetection:
    def test_fna_symbol(self):
        sym = "cutlass::fna::collective::FnaMainloopTmaWarpSpecializedSm90"
        assert _is_fna_symbol(sym) is True

    def test_fmha_symbol(self):
        sym = "cutlass::fmha::collective::FmhaMainloopTmaWarpSpecializedSm90"
        assert _is_fna_symbol(sym) is False

    def test_hopper_fna_in_fmha_wrapper(self):
        # Hopper FNA uses fmha kernel wrapper with fna collective inside
        sym = (
            "cutlass::fmha::kernel::FmhaKernelTmaWarpSpecialized<..., "
            "cutlass::fna::collective::FnaMainloop<...>>"
        )
        assert _is_fna_symbol(sym) is True

    def test_op_name_fna_fwd(self):
        sym = "cutlass::fna::collective::FnaMainloop"
        assert _get_op_name(sym, KernelType.AttentionForward) == "FNAForward"

    def test_op_name_fmha_fwd(self):
        sym = "cutlass::fmha::collective::FmhaMainloop"
        assert _get_op_name(sym, KernelType.AttentionForward) == "FMHAForward"

    def test_op_name_cudnn_fwd(self):
        sym = "cudnn_generated_fort_native_sdpa_sm90_flash_fprop"
        assert _get_op_name(sym, KernelType.AttentionForward) == "FMHAForward"


# ---- Package / Framework / Arch detection ----


class TestPatternMatching:
    """Test the hierarchical AND/OR pattern matcher."""

    def test_simple_string(self):
        assert _match_pattern("hello world", "hello") is True
        assert _match_pattern("hello world", "xyz") is False

    def test_and_tuple(self):
        assert _match_pattern("cudnn flash fprop", ("cudnn", "flash", "fprop")) is True
        assert _match_pattern("cudnn flash bprop", ("cudnn", "flash", "fprop")) is False

    def test_or_nested(self):
        pat = ("cudnn", ("flash", "sdpa", "fmha"), "fprop")
        assert _match_pattern("cudnn_flash_fprop_kernel", pat) is True
        assert _match_pattern("cudnn_sdpa_fprop_kernel", pat) is True
        assert _match_pattern("cudnn_fmha_fprop_kernel", pat) is True
        assert _match_pattern("cudnn_other_fprop_kernel", pat) is False
        assert _match_pattern("cudnn_flash_bprop_kernel", pat) is False


class TestCategoryMapping:
    """Test KernelType -> Category mapping."""

    def test_attn_fwd(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.AttentionForward] == Category.AttnFwd

    def test_attn_bwd_includes_backward(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.AttentionBackward] == Category.AttnBwd

    def test_attn_bwd_includes_sum_odo(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.SumOdO] == Category.AttnBwd

    def test_attn_bwd_includes_convert(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.Convert] == Category.AttnBwd

    def test_memory_ops(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.TokPerm] == Category.MemoryOps

    def test_misc_includes_init(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.Init] == Category.Misc

    def test_misc_includes_misc(self):
        assert KERNEL_TYPE_TO_CATEGORY[KernelType.Misc] == Category.Misc

    def test_all_kernel_types_mapped(self):
        for kt in KernelType:
            assert kt in KERNEL_TYPE_TO_CATEGORY, f"{kt} not in KERNEL_TYPE_TO_CATEGORY"


class TestNamespaceDetection:
    def test_cudnn(self):
        # cuDNN wins over flash when both present
        assert (
            _get_namespace("cudnn_generated_fort_native_sdpa_sm90_flash_fprop")
            == "cuDNN"
        )
        assert _get_namespace("void cudnn::fusion::compute_dot_do_o") == "cuDNN"

    def test_pytorch(self):
        # pytorch_flash matches via "pytorch" before "cutlass" check
        assert _get_namespace("pytorch_flash::flash_fwd<cutlass::half_t>") == "PyTorch"
        assert _get_namespace("at::native::vectorized_elementwise_kernel") == "PyTorch"
        assert _get_namespace("c10::BFloat16_thing") == "PyTorch"

    def test_cutlass(self):
        # cutlass wins over natten when both present
        assert _get_namespace("cutlass::device_kernel_sm90<...>") == "CUTLASS"
        assert (
            _get_namespace(
                "natten::cuda::fna::fna1d_64x64x128_sm80_bfloat16"
                "(natten::cuda::fna::FusedNeighborhoodAttentionKernel<1, cutlass::arch::Sm80>)"
            )
            == "CUTLASS"
        )

    def test_natten(self):
        # natten without cutlass
        assert _get_namespace("natten::utils::something") == "NATTEN"

    def test_flash(self):
        # flash without cudnn
        assert _get_namespace("some_flash_kernel_without_anything_else") == "flash"

    def test_unknown(self):
        assert _get_namespace("random_symbol_noname") == "-"

    def test_arch_sm90(self):
        assert _get_arch("cutlass::device_kernel_sm90<...>") == "Sm90"

    def test_arch_sm80(self):
        assert _get_arch("natten::cuda::fna::fna1d_64x64x128_sm80_bfloat16") == "Sm80"

    def test_arch_none(self):
        assert _get_arch("pytorch_flash::flash_fwd_splitkv_kernel<...>") == "-"


# ---- TensorPool ----


@skip_no_cuda
class TestTensorPool:
    def test_pool_size_small_tensors(self):
        from nattenprof.tensors import _compute_pool_size

        # Small tensors: should get many copies
        shapes = {"q": [1, 64, 1, 64], "k": [1, 64, 1, 64]}
        n = _compute_pool_size(shapes, torch.float16, memory_limit_gb=1.0)
        assert n >= 2
        assert n <= 256

    def test_pool_size_huge_tensors(self):
        from nattenprof.tensors import _compute_pool_size

        # Huge tensors: should get 1 copy
        shapes = {"q": [1, 1000000, 128, 128]}
        n = _compute_pool_size(shapes, torch.float32, memory_limit_gb=1.0)
        assert n == 1

    def test_cycling(self):
        from nattenprof.tensors import TensorPool

        pool = TensorPool(
            shapes={"x": [2, 2]},
            dtype=torch.float32,
            device=torch.device("cuda"),
            memory_limit_gb=0.001,
            seed=42,
            requires_grad=False,
        )
        first = pool.get()
        pool.get()  # advance
        # After cycling through all, should wrap around
        pool.reset()
        first_again = pool.get()
        assert torch.equal(first["x"], first_again["x"])

    def test_deterministic_seed(self):
        from nattenprof.tensors import InitMode, TensorPool

        pool1 = TensorPool(
            shapes={"x": [4, 4]},
            dtype=torch.float32,
            device=torch.device("cuda"),
            init_mode=InitMode.RANDN,
            memory_limit_gb=0.001,
            seed=123,
            requires_grad=False,
        )
        pool2 = TensorPool(
            shapes={"x": [4, 4]},
            dtype=torch.float32,
            device=torch.device("cuda"),
            init_mode=InitMode.RANDN,
            memory_limit_gb=0.001,
            seed=123,
            requires_grad=False,
        )
        assert torch.equal(pool1.get()["x"], pool2.get()["x"])


# ---- Problem format_use_case ----


class TestFormatUseCase:
    def test_na_problem(self):
        p = NAProblem(
            batch_size=2,
            heads=8,
            heads_kv=2,
            dim=128,
            dim_value=64,
            input_size=(256, 256),
            window_size=(80, 80),
            stride=(16, 16),
            dilation=(1, 1),
            is_causal=(False, True),
            dtype=torch.bfloat16,
        )
        s = p.format_use_case(backend="hopper-fna")
        assert "batch=2" in s
        assert "heads=8, heads_kv=2" in s
        assert "dim=128, dim_value=64" in s
        assert "input_size=(256, 256)" in s
        assert "window_size=(80, 80)" in s
        assert "stride=(16, 16)" in s
        assert "dilation=(1, 1)" in s
        assert "is_causal=(False, True)" in s
        assert "dtype=bf16" in s
        assert "backend=hopper-fna" in s

    def test_attn_problem(self):
        p = AttentionProblem(
            batch_size=1,
            heads=4,
            heads_kv=4,
            dim=64,
            dim_value=64,
            seqlen_q=1024,
            seqlen_kv=512,
            dtype=torch.float16,
            is_causal=True,
        )
        s = p.format_use_case(backend="cutlass-fmha")
        assert "seqlen_q=1024" in s
        assert "seqlen_kv=512" in s
        assert "is_causal=True" in s
        assert "dtype=fp16" in s

    def test_na_tensor_shapes_gqa(self):
        p = NAProblem(
            batch_size=2,
            heads=8,
            heads_kv=2,
            dim=128,
            dim_value=64,
            input_size=(32,),
            window_size=(16,),
            stride=(1,),
            dilation=(1,),
            is_causal=(False,),
            dtype=torch.bfloat16,
        )
        shapes = p.get_tensor_shapes()
        assert shapes["q"] == [2, 32, 8, 128]
        assert shapes["k"] == [2, 32, 2, 128]
        assert shapes["v"] == [2, 32, 2, 64]
        assert shapes["d_out"] == [2, 32, 8, 64]

    def test_attn_tensor_shapes_heads_first(self):
        p = AttentionProblem(
            batch_size=1,
            heads=4,
            heads_kv=2,
            dim=64,
            dim_value=32,
            seqlen_q=100,
            seqlen_kv=200,
            dtype=torch.float16,
        )
        shapes = p.get_tensor_shapes(heads_last=False)
        assert shapes["q"] == [1, 4, 100, 64]
        assert shapes["k"] == [1, 2, 200, 64]
        assert shapes["v"] == [1, 2, 200, 32]
