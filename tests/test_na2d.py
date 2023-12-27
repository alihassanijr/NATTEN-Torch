#################################################################################################
# Copyright (c) 2023 Ali Hassani.
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

import logging
import os
import unittest

import torch
from torch.autograd import gradcheck
from torch.utils.cpp_extension import CUDA_HOME

from natten import (disable_gemm_na, disable_tf32, disable_tiled_na,
                    enable_gemm_na, enable_tf32, enable_tiled_na, has_bfloat,
                    has_cuda, has_gemm, has_half)
from natten.functional import natten2dav, natten2dqkrpb

HAS_CUDA = torch.cuda.is_available() and (CUDA_HOME is not None) and has_cuda()
HAS_GEMM = has_gemm()
HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()
logger = logging.getLogger(__name__)


class NA2DTests(unittest.TestCase):
    def _test_against_cpu(
        self, B, H, X, Y, D, kernel_size, dilation, has_bias, dtype, eps
    ):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        with torch.no_grad():
            q, k, v = (
                torch.randn((B, H, X, Y, D)) * (D**-0.5),
                torch.randn((B, H, X, Y, D)),
                torch.randn((B, H, X, Y, D)),
            )
            rpb = None
            if has_bias:
                rpb = torch.randn(H, 2 * kernel_size - 1, 2 * kernel_size - 1)
            q_, k_, v_ = (
                q.clone().cuda().to(dtype),
                k.clone().cuda().to(dtype),
                v.clone().cuda().to(dtype),
            )
            rpb_ = None if rpb is None else rpb.clone().cuda().to(dtype)

            attn_ref = natten2dqkrpb(q, k, rpb, kernel_size, dilation)
            attn_ref = attn_ref.softmax(dim=-1)
            out_ref = natten2dav(attn_ref, v, kernel_size, dilation)

            attn = natten2dqkrpb(q_, k_, rpb_, kernel_size, dilation)
            attn = attn.softmax(dim=-1)
            out = natten2dav(attn, v_, kernel_size, dilation)

            torch.testing.assert_close(attn.float().cpu(), attn_ref, atol=eps, rtol=0)
            torch.testing.assert_close(out.float().cpu(), out_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_cpu(
        self, B, H, X, Y, D, kernel_size, dilation, has_bias=False
    ):
        # Test naive kernels
        disable_tiled_na()
        disable_gemm_na()
        disable_tf32()
        self._test_against_cpu(
            B=B,
            H=H,
            X=X,
            Y=Y,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            dtype=torch.float32,
            eps=1e-4,
        )
        if HAS_HALF:
            self._test_against_cpu(
                B=B,
                H=H,
                X=X,
                Y=Y,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.float16,
                eps=1e-1,
            )
        if HAS_BFLOAT:
            self._test_against_cpu(
                B=B,
                H=H,
                X=X,
                Y=Y,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.bfloat16,
                eps=1e-1,
            )
        if kernel_size < 15:
            # Test tiled kernels
            enable_tiled_na()
            self._test_against_cpu(
                B=B,
                H=H,
                X=X,
                Y=Y,
                D=32,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.float32,
                eps=1e-4,
            )
            if HAS_HALF:
                self._test_against_cpu(
                    B=B,
                    H=H,
                    X=X,
                    Y=Y,
                    D=32,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    has_bias=has_bias,
                    dtype=torch.float16,
                    eps=1e-1,
                )
            if HAS_BFLOAT:
                self._test_against_cpu(
                    B=B,
                    H=H,
                    X=X,
                    Y=Y,
                    D=32,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    has_bias=has_bias,
                    dtype=torch.bfloat16,
                    eps=1e-1,
                )
        # Test GEMM-based kernels
        if HAS_GEMM:
            enable_gemm_na()
            self._test_against_cpu(
                B=B,
                H=H,
                X=X,
                Y=Y,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.float32,
                eps=1e-2,
            )
            enable_tf32()
            self._test_against_cpu(
                B=B,
                H=H,
                X=X,
                Y=Y,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.float32,
                eps=1e-2,
            )
            if HAS_HALF:
                self._test_against_cpu(
                    B=B,
                    H=H,
                    X=X,
                    Y=Y,
                    D=D,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    has_bias=has_bias,
                    dtype=torch.float16,
                    eps=1e-1,
                )
            if HAS_BFLOAT:
                self._test_against_cpu(
                    B=B,
                    H=H,
                    X=X,
                    Y=Y,
                    D=D,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    has_bias=has_bias,
                    dtype=torch.bfloat16,
                    eps=1e-1,
                )

    def test_cpu_vs_cuda(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        torch.manual_seed(42)
        self._test_all_dtypes_against_cpu(
            B=1, H=1, X=16, Y=16, D=32, kernel_size=7, dilation=1
        )
        self._test_all_dtypes_against_cpu(
            B=2, H=1, X=16, Y=16, D=32, kernel_size=7, dilation=1
        )
        self._test_all_dtypes_against_cpu(
            B=2, H=2, X=16, Y=16, D=32, kernel_size=7, dilation=1
        )
        self._test_all_dtypes_against_cpu(
            B=1, H=2, X=28, Y=28, D=32, kernel_size=3, dilation=1
        )
        self._test_all_dtypes_against_cpu(
            B=1, H=2, X=28, Y=28, D=32, kernel_size=3, dilation=4
        )
        self._test_all_dtypes_against_cpu(
            B=1, H=2, X=28, Y=28, D=32, kernel_size=13, dilation=1
        )
        self._test_all_dtypes_against_cpu(
            B=1, H=2, X=28, Y=28, D=32, kernel_size=13, dilation=2
        )
        self._test_all_dtypes_against_cpu(
            B=1, H=2, X=28, Y=28, D=32, has_bias=True, kernel_size=13, dilation=2
        )

    def _test_autograd_qk(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
        torch.manual_seed(42)
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        query = torch.randn((B, H, X, Y, D), **kwargs)
        key = torch.randn((B, H, X, Y, D), **kwargs)
        rpb = torch.randn((H, 2 * kernel_size - 1, 2 * kernel_size - 1), **kwargs)
        variables = [query, key, rpb, kernel_size, dilation]

        assert gradcheck(
            natten2dqkrpb,
            variables,
            eps=1e-6,
            atol=eps,
            rtol=1e-4,
            nondet_tol=0,
            fast_mode=False,
        ), f"Autograd check failed for NA2D: QK."

    def _test_autograd_av(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
        torch.manual_seed(42)
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        attn = torch.randn((B, H, X, Y, kernel_size * kernel_size), **kwargs)
        value = torch.randn((B, H, X, Y, D), **kwargs)
        variables = [attn, value, kernel_size, dilation]

        assert gradcheck(
            natten2dav,
            variables,
            eps=1e-6,
            atol=eps,
            rtol=1e-4,
            nondet_tol=0,
            fast_mode=False,
        ), f"Autograd check failed for NA2D: AV."

    def _test_autograd(self, B, H, X, Y, D, kernel_size, dilation, eps, device):
        self._test_autograd_qk(
            B=B,
            H=H,
            X=X,
            Y=Y,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            device=device,
        )
        self._test_autograd_av(
            B=B,
            H=H,
            X=X,
            Y=Y,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            device=device,
        )

    def test_autograd_cpu(self):
        self._test_autograd(
            B=1, H=1, X=8, Y=7, D=8, kernel_size=5, dilation=1, eps=1e-6, device="cpu"
        )
        self._test_autograd(
            B=1, H=1, X=7, Y=7, D=4, kernel_size=3, dilation=2, eps=1e-6, device="cpu"
        )

    def test_autograd_cuda_naive(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        disable_tiled_na()
        disable_gemm_na()
        disable_tf32()
        self._test_autograd(
            B=1, H=1, X=5, Y=5, D=8, kernel_size=5, dilation=1, eps=1e-6, device="cuda"
        )
        self._test_autograd(
            B=1,
            H=1,
            X=14,
            Y=15,
            D=16,
            kernel_size=7,
            dilation=2,
            eps=1e-6,
            device="cuda",
        )

    def test_autograd_cuda_tiled(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        enable_tiled_na()
        disable_gemm_na()
        disable_tf32()
        self._test_autograd_qk(
            B=1, H=1, X=3, Y=3, D=32, kernel_size=3, dilation=1, eps=1e-6, device="cuda"
        )
        self._test_autograd_qk(
            B=1, H=1, X=5, Y=5, D=32, kernel_size=5, dilation=1, eps=1e-6, device="cuda"
        )
        self._test_autograd_qk(
            B=1,
            H=1,
            X=14,
            Y=14,
            D=32,
            kernel_size=7,
            dilation=2,
            eps=1e-6,
            device="cuda",
        )

    def test_autograd_cuda_gemm(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        if not HAS_GEMM:
            self.skipTest("NATTEN not compiled with GEMM kernels.")
        enable_gemm_na()
        enable_tf32()
        self._test_autograd(
            B=1, H=1, X=8, Y=9, D=8, kernel_size=5, dilation=1, eps=1e-6, device="cuda"
        )
        self._test_autograd(
            B=1, H=1, X=8, Y=7, D=8, kernel_size=7, dilation=1, eps=1e-6, device="cuda"
        )
        self._test_autograd(
            B=1, H=1, X=7, Y=6, D=8, kernel_size=3, dilation=2, eps=1e-6, device="cuda"
        )

    @unittest.expectedFailure
    def test_invalid_kernel_size(self):
        self._test_autograd(
            B=1, H=1, X=8, Y=9, D=8, kernel_size=8, dilation=1, eps=1e-6, device="cuda"
        )

    @unittest.expectedFailure
    def test_invalid_dilation(self):
        self._test_autograd(
            B=1, H=1, X=8, Y=9, D=8, kernel_size=5, dilation=0, eps=1e-6, device="cuda"
        )

    def _test_fwad_qk(self, B, H, X, Y, D, kernel_size, dilation, device):
        torch.manual_seed(42)
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        query = torch.randn((B, H, X, Y, D), **kwargs)
        key = torch.randn((B, H, X, Y, D), **kwargs)

        variables = [query, key, None, kernel_size, dilation]
        assert gradcheck(
            natten2dqkrpb,
            variables,
            check_forward_ad=True,
            check_backward_ad=False,
            check_undefined_grad=False,
            check_batched_grad=False,
            check_grad_dtypes=False,
        ), f"Forward mode autograd check failed for NA2D: QK."

    def _test_fwad_av(self, B, H, X, Y, D, kernel_size, dilation, device):
        torch.manual_seed(42)
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        attn = torch.randn((B, H, X, Y, kernel_size**2), **kwargs)
        value = torch.randn((B, H, X, Y, D), **kwargs)
        variables = [attn, value, kernel_size, dilation]

        assert gradcheck(
            natten2dav,
            variables,
            check_forward_ad=True,
            check_backward_ad=False,
            check_undefined_grad=False,
            check_batched_grad=False,
            check_grad_dtypes=False,
        ), f"Forward mode autograd check failed for NA2D: AV."

    def _test_fwad(self, B, H, X, Y, D, kernel_size, dilation, device):
        self._test_fwad_qk(
            B=B,
            H=H,
            X=X,
            Y=Y,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            device=device,
        )
        self._test_fwad_av(
            B=B,
            H=H,
            X=X,
            Y=Y,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            device=device,
        )

    def test_fwad_cpu(self):
        self._test_fwad(
            B=1, H=1, X=8, Y=7, D=8, kernel_size=5, dilation=1, device="cpu"
        )
        self._test_fwad(
            B=1, H=1, X=7, Y=7, D=4, kernel_size=3, dilation=2, device="cpu"
        )

    def test_fwad_cuda_naive(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        disable_tiled_na()
        disable_gemm_na()
        disable_tf32()
        self._test_fwad(
            B=1, H=1, X=5, Y=5, D=8, kernel_size=5, dilation=1, device="cuda"
        )
        self._test_fwad(
            B=1,
            H=1,
            X=14,
            Y=15,
            D=16,
            kernel_size=7,
            dilation=2,
            device="cuda",
        )

    def test_fwad_cuda_tiled(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        enable_tiled_na()
        disable_gemm_na()
        disable_tf32()
        self._test_fwad_qk(
            B=1, H=1, X=3, Y=3, D=32, kernel_size=3, dilation=1, device="cuda"
        )
        self._test_fwad_qk(
            B=1, H=1, X=5, Y=5, D=32, kernel_size=5, dilation=1, device="cuda"
        )
        self._test_fwad_qk(
            B=1,
            H=1,
            X=14,
            Y=14,
            D=32,
            kernel_size=7,
            dilation=2,
            device="cuda",
        )

    def test_fwad_cuda_gemm(self):
        if not HAS_CUDA:
            self.skipTest("NATTEN not compiled with CUDA.")
        if not HAS_GEMM:
            self.skipTest("NATTEN not compiled with GEMM kernels.")
        enable_gemm_na()
        enable_tf32()
        self._test_fwad(
            B=1, H=1, X=8, Y=9, D=8, kernel_size=5, dilation=1, device="cuda"
        )
        self._test_fwad(
            B=1, H=1, X=8, Y=7, D=8, kernel_size=7, dilation=1, device="cuda"
        )
        self._test_fwad(
            B=1, H=1, X=7, Y=6, D=8, kernel_size=3, dilation=2, device="cuda"
        )


if __name__ == "__main__":
    unittest.main()
