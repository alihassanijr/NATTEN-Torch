#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
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

# This file is the entrypoint for all backend ops.
# The only exception where we directly dlopen should be
# certain unit tests.

from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to load libnatten. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )

from .types import (
    CausalArg1DType,
    CausalArg2DType,
    CausalArg3DType,
    Dimension1DType,
    Dimension2DType,
    Dimension3DType,
    FnaBackwardConfigType,
    FnaForwardConfigType,
    NoneType,
)

def na2d_forward_default(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size: Dimension2DType,
    dilation: Dimension2DType,
    is_causal: CausalArg2DType,
    scale: float,
    q_tile_shape: Dimension2DType,
    kv_tile_shape: Dimension2DType,
):
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    assert bias is None or bias.is_contiguous()
    assert logsumexp is None or logsumexp.is_contiguous()
    output = torch.empty_like(query)
    libnatten.na2d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tile_shape,
        kv_tile_shape,
    )
    return output

# Define a custom op
#
# The schema of the operator says two things:
# 1. what the input/output types are
# 2. If any inputs are being mutated in-place, they must be annotated
#    with (a!). For example, "Tensor(a!) x" says that Tensor x is being mutated
#    in-place.
torch.library.define(
    "natten::na2d_forward",
    "(Tensor query, Tensor key, Tensor value, "
    "Tensor? rel_pos_bias, "
    "Tensor? logsumexp, "
    "int[] kernel_size, "
    "int[] dilation, "
    "bool[] is_causal, "
    "float scale, "
    "int[] q_tile_shape, "
    "int[] kv_tile_shape)"
    " -> Tensor"
)

torch.library.impl("natten::na2d_forward", "default", na2d_forward_default)

@torch.library.impl_abstract("natten::na2d_forward")
def fna_forward_generic_abstract_imp(
    query,
    key,
    value,
    bias,
    logsumexp,
    kernel_size,
    dilation,
    is_causal,
    scale,
    q_tile_shape,
    kv_tile_shape
):
    return torch.empty_like(query)
