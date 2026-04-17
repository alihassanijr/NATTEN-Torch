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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

DTYPE_SHORT = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float8_e4m3fn: "e4m3",
    torch.float8_e5m2: "e5m2",
}


def _format_shape_line(
    batch_size: int,
    heads: int,
    heads_kv: int,
    dim: int,
    dim_value: int,
) -> str:
    return (
        f"batch={batch_size}, "
        f"heads={heads}, heads_kv={heads_kv}, "
        f"dim={dim}, dim_value={dim_value}"
    )


@dataclass
class NAProblem:
    batch_size: int
    heads: int
    heads_kv: int
    dim: int
    dim_value: int
    input_size: Tuple[int, ...]
    window_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    dilation: Tuple[int, ...]
    is_causal: Tuple[bool, ...]
    dtype: torch.dtype
    additional_kv_length: int = 0

    @property
    def na_dim(self) -> int:
        return len(self.input_size)

    @property
    def is_self_attn(self) -> bool:
        return not any(c for c in self.is_causal) and all(
            x == w for x, w in zip(self.input_size, self.window_size)
        )

    def get_tensor_shapes(self) -> Dict[str, List[int]]:
        """Tensor shapes in heads-last layout for NATTEN ops."""
        spatial = list(self.input_size)
        shapes = {
            "q": [self.batch_size] + spatial + [self.heads, self.dim],
            "k": [self.batch_size] + spatial + [self.heads_kv, self.dim],
            "v": [self.batch_size] + spatial + [self.heads_kv, self.dim_value],
            "d_out": [self.batch_size] + spatial + [self.heads, self.dim_value],
        }
        if self.additional_kv_length > 0:
            shapes["add_k"] = [
                self.batch_size,
                self.additional_kv_length,
                self.heads_kv,
                self.dim,
            ]
            shapes["add_v"] = [
                self.batch_size,
                self.additional_kv_length,
                self.heads_kv,
                self.dim_value,
            ]
        return shapes

    def format_use_case(
        self,
        backend: Optional[str] = None,
        fmha_backend: Optional[str] = None,
    ) -> str:
        indent = "  "
        lines = []

        # Line 1: batch, heads, dim
        lines.append(
            _format_shape_line(
                self.batch_size,
                self.heads,
                self.heads_kv,
                self.dim,
                self.dim_value,
            )
        )

        # Line 2: input_size, is_causal
        seq_parts = [
            f"input_size={self.input_size}",
            f"is_causal={self.is_causal}",
        ]
        lines.append(", ".join(seq_parts))

        # Line 3: NA params (window_size, stride, dilation)
        na_parts = [
            f"window_size={self.window_size}",
            f"stride={self.stride}",
            f"dilation={self.dilation}",
        ]
        lines.append(", ".join(na_parts))

        # Line 4: dtype, backend
        backend_parts = [f"dtype={DTYPE_SHORT.get(self.dtype, str(self.dtype))}"]
        if backend is not None:
            backend_parts.append(f"backend={backend}")
        if fmha_backend is not None:
            backend_parts.append(f"fmha_backend={fmha_backend}")
        lines.append(", ".join(backend_parts))

        return "Use case:\n" + "\n".join(indent + line for line in lines)

    def __str__(self) -> str:
        return (
            f"NAProblem("
            f"batch_size={self.batch_size}, "
            f"heads={self.heads}, "
            f"heads_kv={self.heads_kv}, "
            f"dim={self.dim}, "
            f"dim_value={self.dim_value}, "
            f"input_size={self.input_size}, "
            f"window_size={self.window_size}, "
            f"stride={self.stride}, "
            f"dilation={self.dilation}, "
            f"is_causal={self.is_causal}, "
            f"additional_kv_length={self.additional_kv_length}, "
            f"dtype={self.dtype})"
        )


@dataclass
class AttentionProblem:
    batch_size: int
    heads: int
    heads_kv: int
    dim: int
    dim_value: int
    seqlen_q: int
    seqlen_kv: int
    dtype: torch.dtype
    is_causal: bool = False
    seqlens_q: Optional[List[int]] = None
    seqlens_kv: Optional[List[int]] = None

    @property
    def is_varlen(self) -> bool:
        return self.seqlens_q is not None

    def get_tensor_shapes(self, heads_last: bool = True) -> Dict[str, List[int]]:
        """Tensor shapes for attention ops."""
        if self.is_varlen:
            assert self.seqlens_q is not None
            total_q = sum(self.seqlens_q)
            total_kv = sum(self.seqlens_kv) if self.seqlens_kv is not None else total_q
            b = 1
        else:
            total_q = self.seqlen_q
            total_kv = self.seqlen_kv
            b = self.batch_size

        if heads_last:
            return {
                "q": [b, total_q, self.heads, self.dim],
                "k": [b, total_kv, self.heads_kv, self.dim],
                "v": [b, total_kv, self.heads_kv, self.dim_value],
                "d_out": [b, total_q, self.heads, self.dim_value],
            }
        else:
            return {
                "q": [b, self.heads, total_q, self.dim],
                "k": [b, self.heads_kv, total_kv, self.dim],
                "v": [b, self.heads_kv, total_kv, self.dim_value],
                "d_out": [b, self.heads, total_q, self.dim_value],
            }

    def make_varlen_params(self, device: torch.device) -> dict:
        """Creates varlen metadata tensors and scalars."""
        assert self.is_varlen
        assert self.seqlens_q is not None

        seqlens_kv = self.seqlens_kv if self.seqlens_kv is not None else self.seqlens_q

        cum_q = [0]
        for s in self.seqlens_q:
            cum_q.append(cum_q[-1] + s)

        cum_kv = [0]
        for s in seqlens_kv:
            cum_kv.append(cum_kv[-1] + s)

        return {
            "cumulative_seqlen_Q": torch.tensor(
                cum_q, dtype=torch.int32, device=device
            ),
            "cumulative_seqlen_KV": torch.tensor(
                cum_kv, dtype=torch.int32, device=device
            ),
            "max_seqlen_Q": max(self.seqlens_q),
            "max_seqlen_KV": max(seqlens_kv),
        }

    def format_use_case(self, backend: Optional[str] = None) -> str:
        indent = "  "
        lines = []

        # Line 1: batch, heads, dim
        lines.append(
            _format_shape_line(
                self.batch_size,
                self.heads,
                self.heads_kv,
                self.dim,
                self.dim_value,
            )
        )

        # Line 2: seqlens, is_causal
        seq_parts = [
            f"seqlen_q={self.seqlen_q}",
            f"seqlen_kv={self.seqlen_kv}",
            f"is_causal={self.is_causal}",
        ]
        if self.is_varlen:
            seq_parts.append("varlen=True")
        lines.append(", ".join(seq_parts))

        # Line 3: dtype, backend
        backend_parts = [f"dtype={DTYPE_SHORT.get(self.dtype, str(self.dtype))}"]
        if backend is not None:
            backend_parts.append(f"backend={backend}")
        lines.append(", ".join(backend_parts))

        return "Use case:\n" + "\n".join(indent + line for line in lines)

    def __str__(self) -> str:
        return (
            f"AttentionProblem("
            f"batch_size={self.batch_size}, "
            f"heads={self.heads}, "
            f"heads_kv={self.heads_kv}, "
            f"dim={self.dim}, "
            f"dim_value={self.dim_value}, "
            f"seqlen_q={self.seqlen_q}, "
            f"seqlen_kv={self.seqlen_kv}, "
            f"is_causal={self.is_causal}, "
            f"is_varlen={self.is_varlen}, "
            f"dtype={self.dtype})"
        )
