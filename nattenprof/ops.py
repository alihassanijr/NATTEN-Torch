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

"""Thin wrappers that run the actual attention operations.

Each function takes a tensor dict + the problem + **kwargs that are forwarded directly
to the underlying natten/torch function. This way, adding a new knob to
neighborhood_attention_generic / attention / scaled_dot_product_attention requires no
changes here.
"""

from typing import Dict

from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention

from nattenprof.problem import AttentionProblem, NAProblem

SDPA_BACKEND_MAP = {
    "xformers": SDPBackend.EFFICIENT_ATTENTION,
    "fav2": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def run_na(
    tensors: Dict[str, Tensor],
    problem: NAProblem,
    disable_backward: bool = True,
    **kwargs,
):
    """Run neighborhood_attention_generic with kwargs forwarded verbatim."""
    from natten.functional import neighborhood_attention_generic

    # Build attention_kwargs for the inner FMHA path from the same knobs.
    attention_kwargs = {}
    if kwargs.get("fmha_backend") is not None:
        attention_kwargs["backend"] = kwargs["fmha_backend"]
    if kwargs.get("kernel_schedule") is not None:
        attention_kwargs["kernel_schedule"] = kwargs["kernel_schedule"]
    if kwargs.get("torch_compile"):
        attention_kwargs["torch_compile"] = kwargs["torch_compile"]

    out = neighborhood_attention_generic(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        kernel_size=problem.window_size,
        stride=problem.stride,
        dilation=problem.dilation,
        is_causal=problem.is_causal,
        additional_keys=tensors.get("add_k"),
        additional_values=tensors.get("add_v"),
        attention_kwargs=attention_kwargs if attention_kwargs else None,
        **{k: v for k, v in kwargs.items() if k != "fmha_backend"},
    )

    if not disable_backward:
        out.backward(tensors["d_out"])


def run_attn(
    tensors: Dict[str, Tensor],
    problem: AttentionProblem,
    disable_backward: bool = True,
    **kwargs,
):
    """Run natten.attention with kwargs forwarded verbatim."""
    from natten.functional import attention

    if problem.is_varlen:
        params = problem.make_varlen_params(tensors["q"].device)
        kwargs = {
            **kwargs,
            "cumulative_seqlen_Q": params["cumulative_seqlen_Q"],
            "cumulative_seqlen_KV": params["cumulative_seqlen_KV"],
            "max_seqlen_Q": params["max_seqlen_Q"],
            "max_seqlen_KV": params["max_seqlen_KV"],
        }

    out = attention(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        is_causal=problem.is_causal,
        **kwargs,
    )

    if not disable_backward:
        if isinstance(out, tuple):
            out[0].backward(tensors["d_out"])
        else:
            out.backward(tensors["d_out"])


def run_sdpa(
    tensors: Dict[str, Tensor],
    backend: str = "cudnn",
    is_causal: bool = False,
    disable_backward: bool = True,
):
    """Run torch SDPA with a specific backend."""
    if backend not in SDPA_BACKEND_MAP:
        raise ValueError(
            f"Unrecognized SDPA backend '{backend}'. "
            f"Choices: {', '.join(SDPA_BACKEND_MAP.keys())}"
        )

    with sdpa_kernel(backends=[SDPA_BACKEND_MAP[backend]]):
        out = scaled_dot_product_attention(
            tensors["q"], tensors["k"], tensors["v"], is_causal=is_causal
        )

    if not disable_backward:
        out.backward(tensors["d_out"])
