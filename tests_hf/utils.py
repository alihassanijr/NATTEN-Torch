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
#
# Parity testers: run an op through the local `natten` and through the HF-hub kernel with an
# identical backend + tile config on identical inputs, and compare.
#
# Expectation is bit-exact parity (same kernel source, same version) for forward / dK / dV.
# dQ is only bit-exact when NATTEN runs its deterministic backward reduction; with the default
# atomic (kv-parallel) reduction, dQ is not reproducible run-to-run even for one library, so we
# fall back to a relaxed tolerance there. See `dq_atol`.

import contextlib
import math
import os
import random
import tempfile
from typing import List, Optional, Tuple

import natten
import torch
from natten.utils import log

logger = log.get_logger("natten_hf_tests")

# The HF-hub kernel runs in a persistent separate process (see hub_process.py) because two
# libnatten.so cannot co-load. Set once by the session fixture; used to fetch hub configs and run
# hub ops over CUDA IPC.
HUB_PROC = None


def set_hub_process(hp):
    global HUB_PROC
    HUB_PROC = hp


def hub_process():
    assert (
        HUB_PROC is not None
    ), "HubProcess not initialized (see conftest session fixture)"
    return HUB_PROC


# Count of (problem, config) cases skipped because a kernel could not launch on this GPU for the
# forced config. Reported at session end so coverage is never silently reduced.
SKIPPED_LAUNCH_FAIL = {"count": 0}

# Count of (problem, dtype) cases skipped because the backend advertises no configs for them
# (i.e. the backend does not support that problem shape / dtype).
SKIPPED_NO_CONFIG = {"count": 0}

# Count of parity comparisons actually executed (assertions run), per backend. Guards against a
# test "passing" only because every case inside it was skipped.
COMPARED = {}

_LAUNCH_FAIL_MARKER = "Failed to launch"


@contextlib.contextmanager
def _capture_cxx_stderr():
    """Capture C++-level stderr (fd 2) so we can detect libnatten's launch-failure prints.

    libnatten prints "Failed to launch the CUTLASS kernel" and returns uninitialized output
    instead of raising when a forced tile config is not implementable for the problem/arch. We
    redirect fd 2 to a temp file (not a pipe — avoids deadlock on large output) and read it back.
    """
    holder = {"text": ""}
    torch.cuda.synchronize()
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            yield holder
        finally:
            torch.cuda.synchronize()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            holder["text"] = tmp.read().decode(errors="ignore")


# Relaxed dQ tolerance when bit-exactness is not guaranteed (non-deterministic backward, or a
# backend other than cutlass-fna/cutlass-fmha). Do NOT loosen past 1e-1 without a heads-up.
DQ_RELAXED_ATOL = 1e-2

# Backends whose deterministic backward gives bit-exact dQ.
_DQ_EXACT_BACKENDS = ("cutlass-fna", "cutlass-fmha")


def dq_atol(backend: str, deterministic: bool) -> float:
    """Bit-exact dQ only for cutlass-fna/fmha under torch deterministic mode; else relaxed."""
    if deterministic and backend in _DQ_EXACT_BACKENDS:
        return 0.0
    return DQ_RELAXED_ATOL


def reset_everything(seed: int = 42, deterministic: bool = False):
    """Reset RNG + context for the LOCAL natten (parent process).

    The hub runs in its own process and resets its own context per-call (see hub_process). Both
    sides set torch's deterministic flag in their own process to the same value.
    """
    # Set torch's deterministic flag FIRST: use_kv_parallelism_in_fused_na(True) warns+ignores if
    # torch-determinism is on, so it must see the current value, not the previous call's stale one.
    torch.use_deterministic_algorithms(deterministic)
    natten.context.NattenContext.reset()
    natten.set_memory_usage_preference("unrestricted")
    natten.use_kv_parallelism_in_fused_na(not deterministic)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    logger.debug(f"Reset: {seed=}, {deterministic=}")


def _normalize_tile_config(cfg):
    """Config getters return either (q_tile, kv_tile) or ((q_tile, kv_tile), schedule).

    The hopper forms carry a schedule; the worker has stringified it (see hub_process), so the
    nested form is exactly the one whose second element is a str. This works for both FNA (tiles
    are tuples) and FMHA (tiles are ints), where the tuple-vs-int shape alone is ambiguous.

    Returns (q_tile, kv_tile, schedule|None).
    """
    if len(cfg) == 2 and isinstance(cfg[1], str):
        (q_tile, kv_tile), schedule = cfg
        return q_tile, kv_tile, schedule
    q_tile, kv_tile = cfg
    return q_tile, kv_tile, None


def sample_configs(configs: list, n: int, seed_offset: int = 0) -> list:
    """Shuffle and take up to `n` configs (source of truth is the hub's supported set)."""
    configs = list(configs)
    random.shuffle(configs)
    return configs[:n]


def get_backend_configs(backend: str, q, k, v) -> Tuple[list, list]:
    """Return (forward_configs, backward_configs) that the hub advertises for these inputs.

    Enumerated from the hub (source of truth) via the worker process. May be empty if the backend
    does not support this problem/dtype; callers should skip.
    """
    return hub_process().get_configs(backend, q, k, v)


# Per-backend randsweep constraints, mirroring the corresponding generators in tests/. Each
# backend supports a different set of shapes, so a single generic generator would emit shapes the
# backend cannot run. head_dim ranges + whether head_dim_v may differ + 3D size cap all vary.
_FNA_RANDSWEEP = {
    "cutlass-fna": dict(
        head_dim_choices=list(range(8, 193, 8)),
        vary_head_dim_v=True,
        max_size={1: 2**15, 2: 128, 3: 64},
    ),
    "blackwell-fna": dict(
        head_dim_choices=list(range(8, 129, 8)),
        vary_head_dim_v=False,
        max_size={1: 2**15, 2: 128, 3: 96},
    ),
    "hopper-fna": dict(
        head_dim_choices=[32, 64, 128],
        vary_head_dim_v=False,
        max_size={1: 2**15, 2: 128, 3: 96},
    ),
}
_FMHA_RANDSWEEP = {
    "cutlass-fmha": dict(
        head_dim_choices=list(range(8, 1025, 8)),
        supports_gqa=False,
        vary_head_dim_v=True,
    ),
    "blackwell-fmha": dict(
        head_dim_choices=list(range(8, 129, 8)),
        supports_gqa=True,
        vary_head_dim_v=False,
    ),
    "hopper-fmha": dict(
        head_dim_choices=[32, 64, 128], supports_gqa=False, vary_head_dim_v=False
    ),
}


def random_fna_problem(na_dim: int, backend: str, max_seqlen: int = 2**16):
    """Random na{1,2,3}d problem for `backend` (only shapes that backend supports).

    Mirrors the per-backend randsweep in tests/. Call after seeding for reproducibility.
    """
    cfg = _FNA_RANDSWEEP[backend]
    max_size = cfg["max_size"]
    seqlen_limit_batched = 2**13

    input_shape = [random.choice(range(4, max_size[na_dim] + 1)) for _ in range(na_dim)]
    while math.prod(input_shape) > max_seqlen:
        cut = random.choice(range(na_dim))
        input_shape[cut] = max(4, int(input_shape[cut] * 0.1))
    input_shape = tuple(input_shape)

    max_heads = min(max(1, (seqlen_limit_batched // math.prod(input_shape)) * 4), 4)
    heads = random.choice(range(1, max_heads + 1))
    max_batch = min(
        max(1, ((seqlen_limit_batched // math.prod(input_shape)) * 4) // heads), 4
    )
    batch = random.choice(range(1, max_batch + 1))
    heads_kv = random.choice([i for i in range(1, heads + 1) if heads % i == 0])

    head_dim = random.choice(cfg["head_dim_choices"])
    head_dim_v = (
        random.choice(range(max(8, head_dim - 16), min(193, head_dim + 16), 8))
        if cfg["vary_head_dim_v"]
        else head_dim
    )

    kernel_size = tuple(random.choice(range(2, x + 1)) for x in input_shape)
    stride = tuple(random.choice(range(1, k + 1)) for k in kernel_size)
    dilation = tuple(
        random.choice(range(1, x // k + 1)) for x, k in zip(input_shape, kernel_size)
    )
    is_causal = tuple(random.choice([False, True]) for _ in range(na_dim))
    return dict(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )


def random_fmha_problem(backend: str, max_qk: int = 2**20):
    """Random FMHA problem for `backend` (only shapes that backend supports).

    Mirrors the per-backend randsweep in tests/. Call after seeding for reproducibility.
    """
    cfg = _FMHA_RANDSWEEP[backend]
    batch = random.choice(range(1, 4))
    heads = random.choice(range(1, 8 + 1) if cfg["supports_gqa"] else range(1, 4))
    heads_kv = (
        random.choice([i for i in range(1, heads + 1) if heads % i == 0])
        if cfg["supports_gqa"]
        else heads
    )
    head_dim = random.choice(cfg["head_dim_choices"])
    head_dim_v = (
        random.choice(cfg["head_dim_choices"]) if cfg["vary_head_dim_v"] else head_dim
    )

    seqlen_q = random.choice(range(8, 2**13))
    seqlen_kv = random.choice(range(8, 2**13))
    while seqlen_q * seqlen_kv > max_qk:
        if random.choice([True, False]):
            seqlen_kv = max(8, int(seqlen_kv * 0.1))
        else:
            seqlen_q = max(8, int(seqlen_q * 0.1))
    return dict(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
    )


def run_fna_case(
    backend: str,
    batch: int,
    heads: int,
    head_dim: int,
    input_shape: Tuple[int, ...],
    kernel_size,
    stride,
    dilation,
    is_causal,
    dtypes: List[torch.dtype],
    deterministic: bool = False,
    configs_to_test: int = 3,
    heads_kv: Optional[int] = None,
    head_dim_v: Optional[int] = None,
):
    """Full parity case for one na{1,2,3}d problem: sweep dtypes x a few hub configs.

    Returns the number of parity comparisons actually executed.
    """
    compared = 0
    for dtype in dtypes:
        tester = HFFnaParityTester(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            dtype=dtype,
        )
        fwd_cfgs, bwd_cfgs = get_backend_configs(backend, tester.q, tester.k, tester.v)
        if not fwd_cfgs or not bwd_cfgs:
            SKIPPED_NO_CONFIG["count"] += 1
            logger.warning(
                f"SKIP (no configs): {backend} na{len(input_shape)}d shape={input_shape} "
                f"dtype={dtype}"
            )
            continue
        fwd_cfgs = sample_configs(fwd_cfgs, configs_to_test)
        bwd_cfgs = sample_configs(bwd_cfgs, configs_to_test)
        n = max(len(fwd_cfgs), len(bwd_cfgs))
        for i in range(n):
            q_tile, kv_tile, schedule = _normalize_tile_config(
                fwd_cfgs[i % len(fwd_cfgs)]
            )
            bwd_q_tile, bwd_kv_tile, _ = _normalize_tile_config(
                bwd_cfgs[i % len(bwd_cfgs)]
            )
            compared += tester.test(
                backend=backend,
                q_tile=q_tile,
                kv_tile=kv_tile,
                bwd_q_tile=bwd_q_tile,
                bwd_kv_tile=bwd_kv_tile,
                deterministic=deterministic,
                schedule=schedule,
            )
    return compared


def run_fmha_case(
    backend: str,
    batch: int,
    heads: int,
    head_dim: int,
    seqlen_q: int,
    seqlen_kv: int,
    is_causal: bool,
    dtypes: List[torch.dtype],
    deterministic: bool = False,
    configs_to_test: int = 3,
    heads_kv: Optional[int] = None,
    head_dim_v: Optional[int] = None,
):
    """Full parity case for one FMHA problem: sweep dtypes x a few hub configs.

    Returns the number of parity comparisons actually executed.
    """
    compared = 0
    for dtype in dtypes:
        tester = HFFmhaParityTester(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            is_causal=is_causal,
            dtype=dtype,
        )
        fwd_cfgs, bwd_cfgs = get_backend_configs(backend, tester.q, tester.k, tester.v)
        if not fwd_cfgs or not bwd_cfgs:
            SKIPPED_NO_CONFIG["count"] += 1
            logger.warning(
                f"SKIP (no configs): {backend} q={seqlen_q} kv={seqlen_kv} dtype={dtype}"
            )
            continue
        fwd_cfgs = sample_configs(fwd_cfgs, configs_to_test)
        bwd_cfgs = sample_configs(bwd_cfgs, configs_to_test)
        n = max(len(fwd_cfgs), len(bwd_cfgs))
        for i in range(n):
            q_tile, kv_tile, schedule = _normalize_tile_config(
                fwd_cfgs[i % len(fwd_cfgs)]
            )
            bwd_q_tile, bwd_kv_tile, _ = _normalize_tile_config(
                bwd_cfgs[i % len(bwd_cfgs)]
            )
            compared += tester.test(
                backend=backend,
                q_tile=q_tile,
                kv_tile=kv_tile,
                bwd_q_tile=bwd_q_tile,
                bwd_kv_tile=bwd_kv_tile,
                schedule=schedule,
                deterministic=deterministic,
            )
    return compared


class HFFnaParityTester:
    """Generate inputs once, then compare local-natten vs hub for na{1,2,3}d."""

    def __init__(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        input_shape: Tuple[int, ...],
        kernel_size,
        stride,
        dilation,
        is_causal,
        dtype: torch.dtype,
        head_dim_v: Optional[int] = None,
        heads_kv: Optional[int] = None,
    ):
        assert isinstance(input_shape, tuple)
        self.na_dim = len(input_shape)
        assert self.na_dim in (1, 2, 3)

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv or heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        self.dtype = dtype

        with torch.no_grad():
            self.q = torch.randn(
                (batch, *input_shape, self.heads, self.head_dim),
                device="cuda",
                dtype=dtype,
            )
            self.k = torch.randn(
                (batch, *input_shape, self.heads_kv, self.head_dim),
                device="cuda",
                dtype=dtype,
            )
            self.v = torch.randn(
                (batch, *input_shape, self.heads_kv, self.head_dim_v),
                device="cuda",
                dtype=dtype,
            )
            self.d_out = (
                torch.randn(
                    (batch, *input_shape, self.heads, self.head_dim_v),
                    device="cuda",
                    dtype=dtype,
                )
                * 0.05
            )

    def _run_local(
        self, backend, q_tile, kv_tile, bwd_q_tile, bwd_kv_tile, schedule, test_backprop
    ):
        op = {1: natten.na1d, 2: natten.na2d, 3: natten.na3d}[self.na_dim]
        q = self.q.clone().requires_grad_(test_backprop)
        k = self.k.clone().requires_grad_(test_backprop)
        v = self.v.clone().requires_grad_(test_backprop)
        d_out = self.d_out.clone()

        with _capture_cxx_stderr() as cap:
            out = op(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                backend=backend,
                q_tile_shape=q_tile,
                kv_tile_shape=kv_tile,
                backward_q_tile_shape=bwd_q_tile,
                backward_kv_tile_shape=bwd_kv_tile,
                kernel_schedule=schedule,
            )

            dq = dk = dv = None
            if test_backprop:
                out.backward(d_out)
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )
            out = out.data.clone().float()

        launched = _LAUNCH_FAIL_MARKER not in cap["text"]
        return launched, out, dq, dk, dv

    def test(
        self,
        backend: str,
        q_tile,
        kv_tile,
        bwd_q_tile,
        bwd_kv_tile,
        deterministic: bool,
        schedule=None,
        test_backprop: bool = True,
    ):
        logger.debug(
            f"FNA parity {backend} na{self.na_dim}d "
            f"shape={self.input_shape} k={self.kernel_size} s={self.stride} "
            f"d={self.dilation} causal={self.is_causal} dtype={self.dtype} "
            f"det={deterministic} tiles={q_tile}/{kv_tile} bwd={bwd_q_tile}/{bwd_kv_tile} "
            f"sched={schedule}"
        )

        launched_m, out_m, dq_m, dk_m, dv_m = self._run_local(
            backend, q_tile, kv_tile, bwd_q_tile, bwd_kv_tile, schedule, test_backprop
        )

        # hub side: parent-owned shared output buffers, filled in place by the worker via IPC.
        out_h = torch.empty_like(self.d_out)
        dq_h = torch.empty_like(self.q) if test_backprop else None
        dk_h = torch.empty_like(self.k) if test_backprop else None
        dv_h = torch.empty_like(self.v) if test_backprop else None
        status = hub_process().run(
            dict(
                kind="na",
                na_dim=self.na_dim,
                backend=backend,
                test_backprop=test_backprop,
                deterministic=deterministic,
                kwargs=dict(
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=self.dilation,
                    is_causal=self.is_causal,
                    q_tile_shape=q_tile,
                    kv_tile_shape=kv_tile,
                    backward_q_tile_shape=bwd_q_tile,
                    backward_kv_tile_shape=bwd_kv_tile,
                    kernel_schedule=schedule,
                ),
                q=self.q,
                k=self.k,
                v=self.v,
                grad_out=self.d_out,
                out=out_h,
                dq=dq_h,
                dk=dk_h,
                dv=dv_h,
            )
        )

        if not launched_m or status == "launch_fail":
            # Forced config not implementable on this GPU for one/both builds -> not a parity
            # signal. Skip, but count it so reduced coverage is visible.
            SKIPPED_LAUNCH_FAIL["count"] += 1
            logger.warning(
                f"SKIP (launch fail, local={launched_m} hub={status!r}): {backend} "
                f"na{self.na_dim}d shape={self.input_shape} tiles={q_tile}/{kv_tile}"
            )
            return 0

        torch.testing.assert_close(out_h.float(), out_m, atol=0.0, rtol=0.0)
        if test_backprop:
            torch.testing.assert_close(dk_h.float(), dk_m, atol=0.0, rtol=0.0)
            torch.testing.assert_close(dv_h.float(), dv_m, atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                dq_h.float(), dq_m, atol=dq_atol(backend, deterministic), rtol=0.0
            )
        COMPARED[backend] = COMPARED.get(backend, 0) + 1
        return 1


class HFFmhaParityTester:
    """Generate inputs once, then compare local-natten vs hub for `attention` (FMHA)."""

    def __init__(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        dtype: torch.dtype,
        head_dim_v: Optional[int] = None,
        heads_kv: Optional[int] = None,
    ):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv or heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.is_causal = is_causal
        self.dtype = dtype

        with torch.no_grad():
            self.q = torch.randn(
                (batch, seqlen_q, self.heads, self.head_dim), device="cuda", dtype=dtype
            )
            self.k = torch.randn(
                (batch, seqlen_kv, self.heads_kv, self.head_dim),
                device="cuda",
                dtype=dtype,
            )
            self.v = torch.randn(
                (batch, seqlen_kv, self.heads_kv, self.head_dim_v),
                device="cuda",
                dtype=dtype,
            )
            self.d_out = (
                torch.randn(
                    (batch, seqlen_q, self.heads, self.head_dim_v),
                    device="cuda",
                    dtype=dtype,
                )
                * 0.05
            )

    def _run_local(
        self, backend, q_tile, kv_tile, bwd_q_tile, bwd_kv_tile, schedule, test_backprop
    ):
        q = self.q.clone().requires_grad_(test_backprop)
        k = self.k.clone().requires_grad_(test_backprop)
        v = self.v.clone().requires_grad_(test_backprop)
        d_out = self.d_out.clone()

        with _capture_cxx_stderr() as cap:
            out, lse = natten.attention(
                q,
                k,
                v,
                is_causal=self.is_causal,
                backend=backend,
                q_tile_size=q_tile,
                kv_tile_size=kv_tile,
                backward_q_tile_size=bwd_q_tile,
                backward_kv_tile_size=bwd_kv_tile,
                kernel_schedule=schedule,
                return_lse=True,
            )

            dq = dk = dv = None
            if test_backprop:
                out.backward(d_out)
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )
            out = out.data.clone().float()
            lse = lse.clone().float()

        launched = _LAUNCH_FAIL_MARKER not in cap["text"]
        return launched, out, lse, dq, dk, dv

    def test(
        self,
        backend: str,
        q_tile,
        kv_tile,
        bwd_q_tile,
        bwd_kv_tile,
        deterministic: bool,
        schedule=None,
        test_backprop: bool = True,
    ):
        logger.debug(
            f"FMHA parity {backend} "
            f"q={self.seqlen_q} kv={self.seqlen_kv} heads={self.heads}/{self.heads_kv} "
            f"d={self.head_dim}/{self.head_dim_v} causal={self.is_causal} dtype={self.dtype} "
            f"det={deterministic} tiles={q_tile}/{kv_tile} bwd={bwd_q_tile}/{bwd_kv_tile}"
        )

        launched_m, out_m, lse_m, dq_m, dk_m, dv_m = self._run_local(
            backend, q_tile, kv_tile, bwd_q_tile, bwd_kv_tile, schedule, test_backprop
        )

        # hub side: parent-owned shared output buffers, filled in place by the worker via IPC.
        out_h = torch.empty_like(self.d_out)
        lse_h = torch.empty(
            (self.batch, self.seqlen_q, self.heads), device="cuda", dtype=torch.float32
        )
        dq_h = torch.empty_like(self.q) if test_backprop else None
        dk_h = torch.empty_like(self.k) if test_backprop else None
        dv_h = torch.empty_like(self.v) if test_backprop else None
        status = hub_process().run(
            dict(
                kind="attention",
                backend=backend,
                test_backprop=test_backprop,
                deterministic=deterministic,
                kwargs=dict(
                    is_causal=self.is_causal,
                    q_tile_size=q_tile,
                    kv_tile_size=kv_tile,
                    backward_q_tile_size=bwd_q_tile,
                    backward_kv_tile_size=bwd_kv_tile,
                    kernel_schedule=schedule,
                ),
                q=self.q,
                k=self.k,
                v=self.v,
                grad_out=self.d_out,
                out=out_h,
                lse=lse_h,
                dq=dq_h,
                dk=dk_h,
                dv=dv_h,
            )
        )

        if not launched_m or status == "launch_fail":
            SKIPPED_LAUNCH_FAIL["count"] += 1
            logger.warning(
                f"SKIP (launch fail, local={launched_m} hub={status!r}): {backend} "
                f"q={self.seqlen_q} kv={self.seqlen_kv} tiles={q_tile}/{kv_tile}"
            )
            return 0

        torch.testing.assert_close(out_h.float(), out_m, atol=0.0, rtol=0.0)
        torch.testing.assert_close(lse_h.float(), lse_m, atol=0.0, rtol=0.0)
        if test_backprop:
            torch.testing.assert_close(dk_h.float(), dk_m, atol=0.0, rtol=0.0)
            torch.testing.assert_close(dv_h.float(), dv_m, atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                dq_h.float(), dq_m, atol=dq_atol(backend, deterministic), rtol=0.0
            )
        COMPARED[backend] = COMPARED.get(backend, 0) + 1
        return 1
