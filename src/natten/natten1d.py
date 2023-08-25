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
import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from .functional import natten1dav, natten1dqkrpb


class NeighborhoodAttention1D(nn.Module):
    """
    Neighborhood Attention 1D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        dilation=1,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1)))
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv_seq_len=None):
        B, L, C = x.shape
        if L < kernel_size * dilation:
            raise ValueError("Neighborhood attention inputs must be at least of length kernel_size * dilation. "
                             f"Got {self.kernel_size=} and {self.dilation=}, but sequence length is {L}.")
        if kv_seq_len is not None:
            if not isinstance(kv_seq_len, torch.Tensor):
                raise ValueError(f"`kv_seq_len` must be a Tensor, got {type(kv_seq_len)}.")
            if kv_seq_len.dtype != torch.long:
                raise ValueError(f"`kv_seq_len` must a Long (int64) Tensor, got {kv_seq_len.dtype}.")
            min_length = kv_seq_len.min()
            if min_length < kernel_size * dilation:
                raise ValueError("Neighborhood attention inputs must be at least of length kernel_size * dilation. "
                                 f"Got {self.kernel_size=} and {self.dilation=}, but minimum sequence length is {min_length}.")

        qkv = (
            self.qkv(x)
            .reshape(B, L, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten1dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation, kv_seq_len)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten1dav(attn, v, self.kernel_size, self.dilation, kv_seq_len)
        x = x.permute(0, 2, 1, 3).reshape(B, L, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )
