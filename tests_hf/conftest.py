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

import pytest
import torch

from .hub_process import HubProcess
from .utils import logger, set_hub_process


def pytest_configure(config):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")


@pytest.fixture(scope="session", autouse=True)
def hub_worker():
    # Spawn the persistent hub worker ONCE for the whole session (hub kernel lives in its own
    # process; local natten lives here). Reused for every parity comparison.
    if not torch.cuda.is_available():
        yield None
        return
    torch.cuda.init()
    hp = HubProcess("cuda:0")
    set_hub_process(hp)
    try:
        yield hp
    finally:
        set_hub_process(None)
        hp.close()


@pytest.fixture(autouse=True)
def log_test_name(request):
    logger.debug(f"Starting {request.node.name}")
    yield
    logger.debug(f"Finished {request.node.name}")


def pytest_terminal_summary(terminalreporter):
    from .utils import COMPARED, SKIPPED_LAUNCH_FAIL, SKIPPED_NO_CONFIG

    if COMPARED:
        total = sum(COMPARED.values())
        by_backend = ", ".join(f"{b}={c}" for b, c in sorted(COMPARED.items()))
        terminalreporter.write_line(
            f"[tests_hf] {total} parity comparison(s) executed bit-exact: {by_backend}"
        )

    n = SKIPPED_LAUNCH_FAIL["count"]
    if n:
        terminalreporter.write_line(
            f"[tests_hf] {n} (problem, config) case(s) skipped: kernel not implementable on "
            f"this GPU for the forced config (not a parity failure)."
        )
    m = SKIPPED_NO_CONFIG["count"]
    if m:
        terminalreporter.write_line(
            f"[tests_hf] {m} (problem, dtype) case(s) skipped: backend advertises no configs "
            f"(unsupported problem/dtype)."
        )
