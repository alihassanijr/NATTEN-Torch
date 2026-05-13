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

from typing import Any, Callable, Optional, Tuple

import torch

from natten.backends.configs.cutlass_blackwell import (
    check_cutlass_blackwell_fna_backward_config_tensorless,
    check_cutlass_blackwell_fna_forward_config_tensorless,
)
from natten.backends.configs.cutlass_hopper import (
    check_cutlass_hopper_fna_backward_config_tensorless,
    check_cutlass_hopper_fna_forward_config_tensorless,
)
from natten.types import DimensionType, KernelSchedule


def check_varlen_backend_parameters(
    backend: str,
    na_dim: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    run_persistent_kernel: bool = True,  # ignored since all fwd kernels support both schedulers
    kernel_schedule: Optional[KernelSchedule] = None,
) -> Tuple[
    Tuple[DimensionType, DimensionType],
    Tuple[DimensionType, DimensionType],
    dict[str, Any],
]:
    """
    Verifies use case compatibility, and checks performance knobs are valid, and replaces with
    defaults if unspecified. Performance knobs like tile shapes are necessary for initializing
    varlen FNA metadata.

    Parameters:
        backend (str): Backend, either user-specified or automatically selected.

        na_dim (int): Token layout dimensionality.

        head_dim (int): Attention head dimension.

        device (torch.device): Target PyTorch device for runtime.

        dtype (torch.dtype): Tensor element type.

        requires_grad (bool): Whether or not tensors will require backward pass.

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
        forward_tile_shape (tuple): Tuple of forward pass tile shapes for Q and KV.

        backward_tile_shape (tuple): Tuple of backward pass tile shapes for Q and KV.

        extra_kwargs (dict): Additional performance arguments required by the backend.
    """

    if backend not in ["hopper-fna", "blackwell-fna"]:
        raise NotImplementedError(
            f"{backend=} must be added to 'check_varlen_backend_parameters'."
        )

    fwd_checker: Callable = None  # type: ignore[assignment]
    bwd_checker: Callable = None  # type: ignore[assignment]
    if backend == "blackwell-fna":
        fwd_checker = check_cutlass_blackwell_fna_forward_config_tensorless
        bwd_checker = check_cutlass_blackwell_fna_backward_config_tensorless
    elif backend == "hopper-fna":
        fwd_checker = check_cutlass_hopper_fna_forward_config_tensorless
        bwd_checker = check_cutlass_hopper_fna_backward_config_tensorless

    assert fwd_checker is not None
    assert bwd_checker is not None

    fwd_kwargs = {
        "na_dim": na_dim,
        "head_dim": head_dim,
        "dtype": dtype,
        "device": device,
        "q_tile_shape": q_tile_shape,
        "kv_tile_shape": kv_tile_shape,
    }

    if backend == "hopper-fna":
        fwd_kwargs["kernel_schedule"] = kernel_schedule

    fwd_params = fwd_checker(**fwd_kwargs)
    if requires_grad:
        backward_q_tile_shape, backward_kv_tile_shape = bwd_checker(
            na_dim=na_dim,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            q_tile_shape=backward_q_tile_shape,
            kv_tile_shape=backward_kv_tile_shape,
        )
    else:
        backward_q_tile_shape, backward_kv_tile_shape = None, None

    extra_kwargs = {}
    if backend == "hopper-fna":
        (q_tile_shape, kv_tile_shape), kernel_schedule = fwd_params
        extra_kwargs["kernel_schedule"] = kernel_schedule
    elif backend == "blackwell-fna":
        q_tile_shape, kv_tile_shape = fwd_params
    else:
        raise NotImplementedError()

    return (  # type: ignore[return-value]
        (q_tile_shape, kv_tile_shape),
        (backward_q_tile_shape, backward_kv_tile_shape),
        extra_kwargs,
    )
