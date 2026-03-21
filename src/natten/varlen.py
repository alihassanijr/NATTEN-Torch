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

from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from natten.backends import (
    choose_varlen_backend,
    cutlass_blackwell_fna_varlen_generic,
    cutlass_hopper_fna_varlen_generic,
)
from natten.backends.configs.varlen import check_varlen_backend_parameters
from natten.token_permute import generate_fna_varlen_metadata
from natten.types import (
    CausalArgTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
    KernelSchedule,
)
from natten.utils import log
from natten.utils.checks import check_kernel_schedule, fmha_tensor_checks

logger = log.get_logger(__name__)


VariableDimensionType = Optional[List[DimensionType]]


def configure_varlen(
    token_layout_list: List[DimensionType],
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    #
    head_dim_v: Optional[int] = None,
    #
    kernel_size: DimensionTypeOrDed = 2,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    #
    kernel_size_list: VariableDimensionType = None,
    stride_list: VariableDimensionType = None,
    dilation_list: VariableDimensionType = None,
    #
    backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
) -> dict:
    """

    !!! warning
        Variable-Length FNA is an experimental feature and may break if not handled carefully.
        Please open issues if you run into any to help us improve it.
        APIs and the flow can change significantly over time.

    Selects a Variable-Length FNA backend, along with forward and backward configurations, and
    generates metadata necessary to run the operation.
    This function is NOT torch-compilable and must be done ahead of time.

    Parameters:
        token_layout_list (list[tuple]): list of token layouts that describe the various independent
            sets of tokens / sequences in QKV. All elements must be integer tuples of size 1, 2, or
            3, and match each other in size as well.

        head_dim (int): Attention head dimension.

        head_dim_v (Optional[int]): Value/output head dimension (if different from head_dim).

        device (torch.device): Target PyTorch device for runtime.

        dtype (torch.dtype): Tensor element type.

        requires_grad (bool): Whether or not tensors will require backward pass.

        kernel_size (Optional[tuple]): kernel / window size must be provided for verification,
            unless 'kernel_size_list' is provided.

        stride (Optional[tuple]): stride parameter, if used, must be provided for verification,
            unless 'stride_list' is provided.

        dilation (Optional[tuple]): dilation parameter, if used, must be provided for verification,
            unless 'dilation_list' is provided.

        is_causal (Optional[tuple]): is_causal parameter, if used, must be provided for verification.
            This parameter does not support variable values like 'kernel_size', 'stride' and
            'dilation'.

        kernel_size_list (Optional[list[tuple]]): list of kernel sizes corresponding to each token
            layout in 'token_layout_list'. This allows customizing kernel size for varying input
            sizes. If unspecified / None, uses the static 'kernel_size' for the entire batch
            instead.

        stride_list (Optional[list[tuple]]): list of stride values corresponding to each token
            layout in 'token_layout_list'.If unspecified / None, uses the static 'stride' for the
            entire batch instead.

        dilation_list (Optional[list[tuple]]): list of dilation values corresponding to each token
            layout in 'token_layout_list'.If unspecified / None, uses the static 'dilation' for the
            entire batch instead.

    Other Parameters:
        backend (str): Backend implementation to run with. Picks the best available one if
            not specified. Refer to [backends](backends.md) for more information.

        q_tile_shape (tuple): Tile shape for the query token layout in the forward pass kernel.

        kv_tile_shape (tuple): Tile shape for the key-value token layout in the forward pass kernel.

        backward_q_tile_shape (tuple): Tile shape for the query token layout in the backward pass
            kernel.

        backward_kv_tile_shape (tuple): Tile shape for the key/value token layout in the backward
            pass kernel.

        run_persistent_kernel (bool): Whether to use persistent tile scheduling in the forward pass
            kernel. This only applies to the `"blackwell-fna"` backend.

        kernel_schedule (Optional[str]): Kernel type (Hopper architecture only). Choices are
            `None`: pick the default, `"non"` (non-persistent), `"coop"` (warp-specialized
            cooperative), or `"pp"` (warp-specialized ping-ponging). Refer to
            [Hopper FMHA/FNA backend](backends.md#hopper-fna-fmha) for more information.

    Returns:
        metadata (dict): Runtime metadata for the current use case.
    """

    if (
        token_layout_list is None
        or not isinstance(token_layout_list, Sequence)
        or len(token_layout_list) < 1
    ):
        raise ValueError(
            f"token_layout_list must be a non-empty sequence type (i.e. list), got {token_layout_list=}."
        )

    if any(not isinstance(token_layout, tuple) for token_layout in token_layout_list):
        raise ValueError(
            f"token_layout_list must be a list of tuples, got {token_layout_list=}."
        )

    na_dim = len(token_layout_list[0])
    assert na_dim in [1, 2, 3]

    kernel_schedule = check_kernel_schedule(kernel_schedule)

    backend = backend or choose_varlen_backend(
        na_dim=na_dim,
        head_dim=head_dim,
        head_dim_v=head_dim_v or head_dim,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        torch_compile=False,
    )

    (
        (q_tile_shape, kv_tile_shape),
        (backward_q_tile_shape, backward_kv_tile_shape),
        extra_kwargs,
    ) = check_varlen_backend_parameters(
        backend=backend,
        na_dim=na_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        run_persistent_kernel=run_persistent_kernel,
        kernel_schedule=kernel_schedule,
    )

    flip_tiled_dims = backend in ["hopper-fna", "blackwell-fna"]

    metadata = generate_fna_varlen_metadata(
        backend=backend,
        token_layout_list=token_layout_list,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        device=device,
        flip_tiled_dims=flip_tiled_dims,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        #
        kernel_size_list=kernel_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        #
        extra_kwargs=extra_kwargs,
    )

    return metadata


def neighborhood_attention_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    # Varlen-specific args: at least one must be specified
    # Option 1 (preferred): construct 'metadata' once ahead of time and reuse
    metadata: Optional[dict] = None,
    # Option 2 (incompatible with graphs, torch compile): eagerly construct 'metadata' from
    # 'token_layout_list', and the optional 'kernel_size_list', 'stride_list', 'dilation_list' on
    # every call.
    token_layout_list: VariableDimensionType = None,
    #
    kernel_size: DimensionTypeOrDed = 2,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    #
    kernel_size_list: VariableDimensionType = None,
    stride_list: VariableDimensionType = None,
    dilation_list: VariableDimensionType = None,
    #
    scale: Optional[float] = None,
    # Perf-related args
    backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """

    !!! warning
        Variable-Length FNA is an experimental feature and may break if not handled carefully.
        Please open issues if you run into any to help us improve it.
        APIs and the flow can change significantly over time.

    # TODO
    """

    # We use FMHA verifiers here because tensors are sequence-packed
    fmha_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        supports_gqa_mqa=True,
        backend_name="Variable-length Neighborhood Attention",
    )

    if metadata is not None:
        assert backend is None or backend == metadata["backend"]
    else:
        if (
            token_layout_list is None
            or not isinstance(token_layout_list, Sequence)
            or len(token_layout_list) < 1
        ):
            raise ValueError(
                f"token_layout_list must be a non-empty sequence type (i.e. list), got {token_layout_list=}."
            )

        if any(
            not isinstance(token_layout, tuple) for token_layout in token_layout_list
        ):
            raise ValueError(
                f"token_layout_list must be a list of tuples, got {token_layout_list=}."
            )

        na_dim = len(token_layout_list[0])

        assert na_dim in [1, 2, 3]

        metadata = configure_varlen(
            token_layout_list=token_layout_list,
            head_dim=query.shape[-1],
            head_dim_v=value.shape[-1],
            device=query.device,
            dtype=query.dtype,
            requires_grad=query.requires_grad,
            #
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            #
            kernel_size_list=kernel_size_list,
            stride_list=stride_list,
            dilation_list=dilation_list,
            #
            backend=backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
        )
        backend = metadata["backend"]

    assert backend is not None
    assert metadata is not None

    # kernel_size, stride, dilation, and is_causal are verified in
    # generate_fna_varlen_metadata

    scale = scale or query.shape[-1] ** -0.5

    if backend == "blackwell-fna":
        outputs = cutlass_blackwell_fna_varlen_generic(
            query=query,
            key=key,
            value=value,
            metadata=metadata,
            scale=scale,
            run_persistent_kernel=run_persistent_kernel,
            return_lse=return_lse,
        )

    elif backend == "hopper-fna":
        outputs = cutlass_hopper_fna_varlen_generic(
            query=query,
            key=key,
            value=value,
            metadata=metadata,
            scale=scale,
            return_lse=return_lse,
        )

    else:
        raise NotImplementedError(f"Backend {backend} does not implement varlen FNA.")

    return outputs
