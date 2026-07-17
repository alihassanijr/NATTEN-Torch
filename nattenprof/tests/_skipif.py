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

"""Shared pytest skip markers. Thin wrappers around natten's primitives
(`get_device_cc`, `_IS_CUDA_AVAILABLE`, `HAS_LIBNATTEN`) — natten's
`skip_if_*_kernels_not_supported` decorators are unittest-shaped and don't
plug into pytest markers, so we bind the same conditions here once.
"""

import pytest
from natten._environment import _IS_CUDA_AVAILABLE, HAS_LIBNATTEN
from natten.utils.device import get_device_cc


def _cc() -> int:
    return get_device_cc() if _IS_CUDA_AVAILABLE else 0


skip_no_cuda = pytest.mark.skipif(not _IS_CUDA_AVAILABLE, reason="CUDA not available")
skip_no_libnatten = pytest.mark.skipif(
    not _IS_CUDA_AVAILABLE or not HAS_LIBNATTEN,
    reason="Requires libnatten + CUDA",
)
skip_no_hopper = pytest.mark.skipif(
    not _IS_CUDA_AVAILABLE or _cc() != 90,
    reason="Hopper kernels require SM90",
)
skip_no_blackwell = pytest.mark.skipif(
    not _IS_CUDA_AVAILABLE or _cc() not in (100, 103),
    reason="Blackwell kernels require SM100 or SM103",
)
