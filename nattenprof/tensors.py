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

import math
from enum import Enum
from typing import Dict, List

import torch
from torch import Tensor


class InitMode(Enum):
    RANDN = "randn"
    UNIFORM = "uniform"
    ONES = "ones"


from natten.utils.dtype import is_fp8 as _is_fp8


def _safe_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float16 if _is_fp8(dtype) else dtype


def _compute_pool_size(
    shapes: Dict[str, List[int]],
    dtype: torch.dtype,
    memory_limit_gb: float,
) -> int:
    safe = _safe_dtype(dtype)
    element_size = torch.tensor([], dtype=safe).element_size()
    one_set_bytes = sum(math.prod(shape) * element_size for shape in shapes.values())
    one_set_gb = one_set_bytes / (1024**3)

    if one_set_gb >= memory_limit_gb:
        return 1

    n = int(memory_limit_gb / one_set_gb)
    return max(2, min(n, 256))


class TensorPool:
    """Pre-allocates N copies of input tensor sets, cycles through them.

    Avoids L2 cache reuse artifacts by ensuring different memory is touched
    on each profiling iteration.
    """

    def __init__(
        self,
        shapes: Dict[str, List[int]],
        dtype: torch.dtype,
        device: torch.device,
        init_mode: InitMode = InitMode.RANDN,
        memory_limit_gb: float = 10.0,
        seed: int = 42,
        requires_grad: bool = False,
    ):
        self.shapes = shapes
        self.dtype = dtype
        self.device = device
        self.init_mode = init_mode
        self.seed = seed
        self.requires_grad = requires_grad
        self._safe_dtype = _safe_dtype(dtype)
        self._needs_typecast = self._safe_dtype != dtype
        self._pool_size = _compute_pool_size(shapes, dtype, memory_limit_gb)
        self._pool: List[Dict[str, Tensor]] = []
        self._index = 0
        self._generate_tensors()

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def _generate_tensors(self):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)

        for _ in range(self._pool_size):
            tensor_set = {}
            for name, shape in self.shapes.items():
                if self.init_mode == InitMode.RANDN:
                    t = torch.randn(
                        shape,
                        dtype=self._safe_dtype,
                        device=self.device,
                        generator=gen,
                    )
                elif self.init_mode == InitMode.UNIFORM:
                    t = torch.rand(
                        shape,
                        dtype=self._safe_dtype,
                        device=self.device,
                        generator=gen,
                    )
                elif self.init_mode == InitMode.ONES:
                    t = torch.ones(
                        shape,
                        dtype=self._safe_dtype,
                        device=self.device,
                    )
                else:
                    raise ValueError(f"Unknown init mode: {self.init_mode}")

                if self._needs_typecast:
                    t = t.to(self.dtype)

                tensor_set[name] = t
            self._pool.append(tensor_set)

    def get(self) -> Dict[str, Tensor]:
        """Return next tensor set, advance counter, wrap around.

        For backward-pass profiling: clears accumulated grads and re-enables
        requires_grad in-place, avoiding clones that would defeat L2 diversity.
        """
        tensor_set = self._pool[self._index]
        self._index = (self._index + 1) % self._pool_size

        if self.requires_grad:
            for t in tensor_set.values():
                if t.grad is not None:
                    t.grad = None
                t.requires_grad_(True)

        return tensor_set

    def reset(self):
        """Reset cycling counter to 0. Does not re-generate tensors."""
        self._index = 0
