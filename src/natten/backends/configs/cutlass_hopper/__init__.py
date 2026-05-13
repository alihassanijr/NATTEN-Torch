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

from typing import List, Optional

import torch  # noqa: F401
from torch import Tensor

from natten.types import (
    CutlassHopperFmhaBackwardConfigType,
    CutlassHopperFmhaForwardConfigType,
    CutlassHopperFnaBackwardConfigType,
    CutlassHopperFnaForwardConfigType,
    DimensionType,
    KernelSchedule,
)
from natten.utils.checks import check_tile_shape
from natten.utils.device import get_device_cc

SUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
]

DTYPE_TO_BITS = {
    torch.float16: 16,
    torch.bfloat16: 16,
}

SUPPORTED_HEAD_DIMS_FWD = [32, 64, 128, 256]
SUPPORTED_HEAD_DIMS_BWD = [32, 64, 128]

# TODO: notes

HOPPER_FORWARD_CONFIGS = {
    1: {
        16: {
            32: [
                (((64,), (128,)), KernelSchedule.NonPersistent),
            ],
            64: [
                (((64,), (128,)), KernelSchedule.NonPersistent),
            ],
            128: [
                (((128,), (128,)), KernelSchedule.WarpSpecializedCooperative),
                (((128,), (128,)), KernelSchedule.WarpSpecializedPingpong),
            ],
            256: [
                (((128,), (64,)), KernelSchedule.WarpSpecializedCooperative),
            ],
        },
    },
    2: {
        16: {
            32: [
                (((8, 8), (16, 8)), KernelSchedule.NonPersistent),
                (((8, 8), (8, 16)), KernelSchedule.NonPersistent),
            ],
            64: [
                (((8, 8), (16, 8)), KernelSchedule.NonPersistent),
                (((8, 8), (8, 16)), KernelSchedule.NonPersistent),
            ],
            128: [
                (((16, 8), (16, 8)), KernelSchedule.WarpSpecializedCooperative),
                (((16, 8), (16, 8)), KernelSchedule.WarpSpecializedPingpong),
            ],
            256: [
                (((16, 8), (8, 8)), KernelSchedule.WarpSpecializedCooperative),
                (((8, 16), (8, 8)), KernelSchedule.WarpSpecializedCooperative),
            ],
        },
    },
    3: {
        16: {
            32: [
                (((4, 4, 4), (4, 4, 8)), KernelSchedule.NonPersistent),
                (((4, 4, 4), (2, 8, 8)), KernelSchedule.NonPersistent),
            ],
            64: [
                (((4, 4, 4), (4, 4, 8)), KernelSchedule.NonPersistent),
                (((4, 4, 4), (2, 8, 8)), KernelSchedule.NonPersistent),
            ],
            128: [
                (((4, 4, 8), (4, 4, 8)), KernelSchedule.WarpSpecializedCooperative),
                (((4, 4, 8), (4, 4, 8)), KernelSchedule.WarpSpecializedPingpong),
                (((2, 8, 8), (2, 8, 8)), KernelSchedule.WarpSpecializedCooperative),
                (((2, 8, 8), (2, 8, 8)), KernelSchedule.WarpSpecializedPingpong),
            ],
            256: [
                (((4, 4, 8), (4, 4, 4)), KernelSchedule.WarpSpecializedCooperative),
                (((2, 8, 8), (4, 4, 4)), KernelSchedule.WarpSpecializedCooperative),
            ],
        },
    },
}

HOPPER_BACKWARD_CONFIGS = {
    1: {
        16: {
            32: [
                ((64,), (128,)),
                ((128,), (128,)),
            ],
            64: [
                ((64,), (128,)),
                ((128,), (128,)),
            ],
            128: [
                ((64,), (128,)),
            ],
        },
    },
    2: {
        16: {
            32: [
                ((8, 8), (16, 8)),
                ((8, 8), (8, 16)),
                ((16, 8), (16, 8)),
                ((16, 8), (8, 16)),
            ],
            64: [
                ((8, 8), (16, 8)),
                ((8, 8), (8, 16)),
                ((16, 8), (16, 8)),
                ((16, 8), (8, 16)),
            ],
            128: [
                ((8, 8), (16, 8)),
                ((8, 8), (8, 16)),
            ],
        },
    },
    3: {
        16: {
            32: [
                ((4, 4, 4), (4, 4, 8)),
                ((4, 4, 4), (2, 8, 8)),
                ((4, 4, 8), (4, 4, 8)),
                ((4, 4, 8), (2, 8, 8)),
            ],
            64: [
                ((4, 4, 4), (4, 4, 8)),
                ((4, 4, 4), (2, 8, 8)),
                ((4, 4, 8), (4, 4, 8)),
                ((4, 4, 8), (2, 8, 8)),
            ],
            128: [
                ((4, 4, 4), (4, 4, 8)),
                ((4, 4, 4), (2, 8, 8)),
                ((2, 4, 8), (2, 8, 8)),
                ((1, 8, 8), (2, 8, 8)),
            ],
        },
    },
}


# NOTE (ali again in 2026): for various reasons, we want to slowly move away from mapping Tensors to
# configs/backends, and request specs directly (na_dim, head_dim, dtype, device).
# We're making all the base APIs "tensor-less" and slowly replacing the old APIs with the new
# format.


# FNA API
def get_all_forward_configs(
    na_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> List[CutlassHopperFnaForwardConfigType]:
    assert na_dim in [1, 2, 3]
    device_cc = get_device_cc(device)
    if (
        device_cc != 90
        or head_dim not in SUPPORTED_HEAD_DIMS_FWD
        or dtype not in SUPPORTED_DTYPES
    ):
        return []

    dtype_bits = DTYPE_TO_BITS[dtype]
    assert dtype_bits in HOPPER_FORWARD_CONFIGS[na_dim]  # type: ignore
    assert head_dim in HOPPER_FORWARD_CONFIGS[na_dim][dtype_bits]  # type: ignore

    return HOPPER_FORWARD_CONFIGS[na_dim][dtype_bits][head_dim]  # type: ignore


def get_all_backward_configs(
    na_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> List[CutlassHopperFnaBackwardConfigType]:
    assert na_dim in [1, 2, 3]
    device_cc = get_device_cc(device)
    if (
        device_cc != 90
        or head_dim not in SUPPORTED_HEAD_DIMS_BWD
        or dtype not in SUPPORTED_DTYPES
    ):
        return []

    dtype_bits = DTYPE_TO_BITS[dtype]
    assert dtype_bits in HOPPER_BACKWARD_CONFIGS[na_dim]  # type: ignore
    assert head_dim in HOPPER_BACKWARD_CONFIGS[na_dim][dtype_bits]  # type: ignore

    return HOPPER_BACKWARD_CONFIGS[na_dim][dtype_bits][head_dim]  # type: ignore


# FMHA API
def get_all_fmha_forward_configs(
    input_tensor: Tensor,
) -> List[CutlassHopperFmhaForwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    configs_multi_dim = get_all_forward_configs(
        na_dim=1,
        head_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    assert all(len(q_t) == len(kv_t) == 1 for (q_t, kv_t), _ in configs_multi_dim)

    configs_fmha = [
        ((q_t[0], kv_t[0]), sched) for (q_t, kv_t), sched in configs_multi_dim
    ]

    return configs_fmha


# FMHA API
def get_all_fmha_backward_configs(
    input_tensor: Tensor,
) -> List[CutlassHopperFmhaBackwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    configs_multi_dim = get_all_backward_configs(
        na_dim=1,
        head_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in configs_multi_dim)

    configs_fmha = [(q_t[0], kv_t[0]) for q_t, kv_t in configs_multi_dim]

    return configs_fmha


# FNA API
def get_default_forward_config(
    input_tensor: Tensor,
) -> CutlassHopperFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return get_default_forward_config_tensorless(
        na_dim=na_dim,
        head_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


# FNA API - tensor-less
def get_default_forward_config_tensorless(
    na_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CutlassHopperFnaForwardConfigType:
    all_configs = get_all_forward_configs(
        na_dim=na_dim,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    if len(all_configs) < 1:
        device_cc = get_device_cc(device)
        raise ValueError(
            "No configs exist for this use case; Hopper FMHA/FNA does not support it: "
            f"{na_dim=}, {head_dim=}, {dtype=}, {device=} ({device_cc=})."
        )

    return all_configs[0]


# FMHA API
def get_default_fmha_forward_config(
    input_tensor: Tensor,
) -> CutlassHopperFmhaForwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    (q_t, kv_t), sched = get_default_forward_config(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return (q_t[0], kv_t[0]), sched


# FNA API
def get_default_backward_config(
    input_tensor: Tensor,
) -> CutlassHopperFnaBackwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return get_default_backward_config_tensorless(
        na_dim=na_dim,
        head_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )


# FNA API - tensor-less
def get_default_backward_config_tensorless(
    na_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CutlassHopperFnaBackwardConfigType:
    all_configs = get_all_backward_configs(
        na_dim=na_dim,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    if len(all_configs) < 1:
        device_cc = get_device_cc(device)
        raise ValueError(
            "No configs exist for this use case; Hopper FMHA/FNA does not support it: "
            f"{na_dim=}, {head_dim=}, {dtype=}, {device=} ({device_cc=})."
        )

    return all_configs[0]


# FMHA API
def get_default_fmha_backward_config(
    input_tensor: Tensor,
) -> CutlassHopperFmhaBackwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    q_t, kv_t = get_default_backward_config(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return q_t[0], kv_t[0]


# FNA API
def check_cutlass_hopper_fna_forward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    kernel_schedule: Optional[KernelSchedule] = None,
) -> CutlassHopperFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return check_cutlass_hopper_fna_forward_config_tensorless(
        na_dim=na_dim,
        head_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        kernel_schedule=kernel_schedule,
    )


# FNA API - tensor-less
def check_cutlass_hopper_fna_forward_config_tensorless(
    na_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    kernel_schedule: Optional[KernelSchedule] = None,
) -> CutlassHopperFnaForwardConfigType:
    assert na_dim in [1, 2, 3]

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    (default_q_tile_shape, default_kv_tile_shape), default_sched = (
        get_default_forward_config_tensorless(
            na_dim=na_dim, head_dim=head_dim, dtype=dtype, device=device
        )
    )

    if q_tile_shape is None and kv_tile_shape is None and kernel_schedule is None:
        return (default_q_tile_shape, default_kv_tile_shape), default_sched  # type: ignore[return-value]

    elif q_tile_shape is None and kv_tile_shape is None:
        q_tile_shape = default_q_tile_shape
        kv_tile_shape = default_kv_tile_shape

    q_tile_shape = check_tile_shape(q_tile_shape)
    kv_tile_shape = check_tile_shape(kv_tile_shape)

    configs = get_all_forward_configs(
        na_dim=na_dim,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    for (q_t, kv_t), sched in configs:
        if (
            q_t == q_tile_shape
            and kv_t == kv_tile_shape
            and (kernel_schedule is None or sched == kernel_schedule)
        ):
            return (q_t, kv_t), sched  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, ((q_t, kv_t), sched) in enumerate(configs):
        examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}, schedule={sched}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Hopper FNA-{na_dim}D. "
        f"Q tile shape {q_tile_shape}, KV tile shape {kv_tile_shape}, and schedule {kernel_schedule} "
        f"are not among the {len(configs)} configurations implementable "
        f"with CUTLASS Hopper FNA. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_hopper_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


# FMHA API
def check_cutlass_hopper_fmha_forward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    kernel_schedule: Optional[KernelSchedule] = None,
) -> CutlassHopperFmhaForwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    (default_q_tile_size, default_kv_tile_size), default_sched = (
        get_default_fmha_forward_config(input_tensor=input_tensor)
    )
    if q_tile_size is None and kv_tile_size is None and kernel_schedule is None:
        return (default_q_tile_size, default_kv_tile_size), default_sched

    elif q_tile_size is None and kv_tile_size is None:
        q_tile_size = default_q_tile_size
        kv_tile_size = default_kv_tile_size

    configs = get_all_fmha_forward_configs(input_tensor=input_tensor)

    for (q_t, kv_t), sched in configs:
        if (
            q_t == q_tile_size
            and kv_t == kv_tile_size
            and (kernel_schedule is None or sched == kernel_schedule)
        ):
            return (q_t, kv_t), sched  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, ((q_t, kv_t), sched) in enumerate(configs):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}, schedule={sched}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Hopper FMHA. "
        f"Q tile size {q_tile_size}, KV tile size {kv_tile_size}, and schedule {kernel_schedule} "
        f"are not among the {len(configs)} configurations implementable "
        f"with CUTLASS Hopper FMHA. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_hopper_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


# FNA API
def check_cutlass_hopper_fna_backward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> CutlassHopperFnaBackwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return check_cutlass_hopper_fna_backward_config_tensorless(
        na_dim=na_dim,
        head_dim=input_tensor.shape[-1],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
    )


# FNA API - tensor-less
def check_cutlass_hopper_fna_backward_config_tensorless(
    na_dim: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> CutlassHopperFnaBackwardConfigType:
    assert na_dim in [1, 2, 3]

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    default_q_tile_shape, default_kv_tile_shape = (
        get_default_backward_config_tensorless(
            na_dim=na_dim, head_dim=head_dim, dtype=dtype, device=device
        )
    )
    if q_tile_shape is None and kv_tile_shape is None:
        return default_q_tile_shape, default_kv_tile_shape  # type: ignore[return-value]

    elif q_tile_shape is None and kv_tile_shape is None:
        q_tile_shape = default_q_tile_shape
        kv_tile_shape = default_kv_tile_shape

    q_tile_shape = check_tile_shape(q_tile_shape)
    kv_tile_shape = check_tile_shape(kv_tile_shape)

    configs = get_all_backward_configs(
        na_dim=na_dim,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    for q_t, kv_t in configs:
        if q_t == q_tile_shape and kv_t == kv_tile_shape:
            return q_t, kv_t  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(configs):
        examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Hopper FNA-{na_dim}D Backward. "
        f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
        f"are not among the {len(configs)} configurations implementable "
        f"with CUTLASS Hopper FNA Backward. "
        "Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_hopper_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


# FMHA API
def check_cutlass_hopper_fmha_backward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> CutlassHopperFmhaBackwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    default_q_tile_size, default_kv_tile_size = get_default_fmha_backward_config(
        input_tensor=input_tensor
    )
    if q_tile_size is None and kv_tile_size is None:
        return default_q_tile_size, default_kv_tile_size

    elif q_tile_size is None and kv_tile_size is None:
        q_tile_size = default_q_tile_size
        kv_tile_size = default_kv_tile_size

    configs = get_all_fmha_backward_configs(input_tensor=input_tensor)

    for q_t, kv_t in configs:
        if q_t == q_tile_size and kv_t == kv_tile_size:
            return q_t, kv_t  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(configs):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Hopper FMHA Backward. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(configs)} configurations implementable "
        f"with CUTLASS Hopper FMHA Backward. "
        "Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_hopper_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )
