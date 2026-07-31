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
# Run the HF-hub NATTEN kernel in a PERSISTENT separate process, sharing CUDA tensors with the
# parent via CUDA IPC.
#
# Two libnatten.so cannot coexist in one process: CUDA APIs like
# cudaFuncSetAttribute(kernel, MaxDynamicSharedMemorySize, ...) key off the host kernel-function
# stub, and with two fatbins registering identically-named CUTLASS sm100 kernels the driver routes
# to the wrong module -> the dynamic-smem opt-in is lost -> "Failed to launch". So the hub kernel
# runs here in its own process (hub loaded, local `natten` NOT), and the parent (local `natten`,
# no hub) drives it over queues, passing GPU buffers by CUDA IPC.
#
# The worker is spawned ONCE and reused for every call (persistent loop on a queue).
#
# IMPORTANT: this module MUST NOT import `natten` (directly or transitively). It is re-imported in
# the spawned child; importing natten there would recreate the very co-load collision we avoid.

import os
import tempfile

import torch
import torch.multiprocessing as mp

_HUB_REPO = "kernels-staging/natten"
_HUB_VERSION = 0
_LAUNCH_FAIL_MARKER = "Failed to launch"

# backend -> (fwd getter, bwd getter) attribute names on the hub module.
_GETTERS = {
    "cutlass-fna": ("get_configs_for_cutlass_fna", "get_bwd_configs_for_cutlass_fna"),
    "hopper-fna": (
        "get_configs_for_cutlass_hopper_fna",
        "get_bwd_configs_for_cutlass_hopper_fna",
    ),
    "blackwell-fna": (
        "get_configs_for_cutlass_blackwell_fna",
        "get_bwd_configs_for_cutlass_blackwell_fna",
    ),
    "cutlass-fmha": (
        "get_configs_for_cutlass_fmha",
        "get_bwd_configs_for_cutlass_fmha",
    ),
    "hopper-fmha": (
        "get_configs_for_cutlass_hopper_fmha",
        "get_bwd_configs_for_cutlass_hopper_fmha",
    ),
    "blackwell-fmha": (
        "get_configs_for_cutlass_blackwell_fmha",
        "get_bwd_configs_for_cutlass_blackwell_fmha",
    ),
}


def _load_hub():
    from kernels import get_kernel

    return get_kernel(_HUB_REPO, version=_HUB_VERSION, trust_remote_code=True)


def _capture_cxx_stderr(fn):
    """Run fn() capturing C++ fd-2 output; return (result, captured_text)."""
    torch.cuda.synchronize()
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            result = fn()
            torch.cuda.synchronize()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            text = tmp.read().decode(errors="ignore")
    return result, text


def _reset_hub(hub, deterministic):
    # Set torch's deterministic flag FIRST: use_kv_parallelism_in_fused_na(True) warns+ignores if
    # torch-determinism is on, so it must see the current value, not the previous task's stale one.
    torch.use_deterministic_algorithms(deterministic)
    hub.context.NattenContext.reset()
    hub.set_memory_usage_preference("unrestricted")
    hub.use_kv_parallelism_in_fused_na(not deterministic)


def _run_one(hub, task):
    """Execute one hub op on the shared tensors, writing results into the shared out buffers."""
    kind = task["kind"]
    backend = task["backend"]
    bp = task["test_backprop"]
    kw = dict(task["kwargs"])

    _reset_hub(hub, task["deterministic"])

    # Clone shared inputs to leaf tensors (values identical to parent's shared buffers).
    q = task["q"].clone().requires_grad_(bp)
    k = task["k"].clone().requires_grad_(bp)
    v = task["v"].clone().requires_grad_(bp)

    def call():
        if kind == "attention":
            return hub.attention(q, k, v, backend=backend, return_lse=True, **kw)
        fn = {1: hub.na1d, 2: hub.na2d, 3: hub.na3d}[task["na_dim"]]
        return fn(q, k, v, backend=backend, **kw)

    out, text = _capture_cxx_stderr(call)
    if _LAUNCH_FAIL_MARKER in text:
        return "launch_fail"

    if kind == "attention":
        out_t, lse_t = out
        task["lse"].copy_(lse_t.detach())
    else:
        out_t = out
    task["out"].copy_(out_t.detach())

    if bp:
        out_t.backward(task["grad_out"])
        task["dq"].copy_(q.grad)
        task["dk"].copy_(k.grad)
        task["dv"].copy_(v.grad)

    # Backward + the copy_ into the shared buffers run AFTER the forward-only capture sync above.
    # Must fully complete before we signal "ok", or the parent reads the buffers mid-flight.
    torch.cuda.synchronize()
    return "ok"


# Hopper FORWARD configs carry a KernelSchedule enum from the hub module. That enum's class does
# not exist in the parent process, so it cannot be unpickled across the queue; and the parent's
# local natten wants its own schedule type anyway. natten accepts a str kernel_schedule, so convert
# enum -> "non"/"coop"/"pp" before crossing the boundary. (Backward configs carry no schedule.)
_SCHED_NAME_TO_STR = {
    "NonPersistent": "non",
    "WarpSpecializedCooperative": "coop",
    "WarpSpecializedPingpong": "pp",
}


def _sanitize_fwd_config(cfg):
    # hopper fwd: ((q_tile, kv_tile), schedule_enum); other fwd configs are plain int/tuple tiles.
    if len(cfg) == 2 and not isinstance(cfg[1], (int, tuple)):
        return (cfg[0], _SCHED_NAME_TO_STR[cfg[1].name])
    return cfg


def _get_configs(hub, task):
    """Return (fwd_configs, bwd_configs) the hub advertises for these shared inputs."""
    fwd_name, bwd_name = _GETTERS[task["backend"]]
    q, k, v = task["q"], task["k"], task["v"]
    fwd = [_sanitize_fwd_config(c) for c in getattr(hub, fwd_name)(q, k, v)]
    bwd = list(getattr(hub, bwd_name)(q, k, v))
    return fwd, bwd


def _worker_main(task_q, res_q, device):
    torch.cuda.set_device(device)
    try:
        hub = _load_hub()
    except Exception as e:  # noqa: BLE001
        res_q.put(("init_error", repr(e)))
        return
    res_q.put(("ready", None))

    while True:
        task = task_q.get()
        if task is None:
            return
        try:
            if task["cmd"] == "configs":
                res_q.put(("ok", _get_configs(hub, task)))
            else:
                res_q.put((_run_one(hub, task), None))
        except Exception as e:  # noqa: BLE001
            res_q.put(("error", repr(e)))


class HubProcess:
    """Persistent subprocess running the HF-hub NATTEN kernel; shares CUDA tensors via IPC.

    Spawned once, reused for every `run(...)`. Parent allocates all buffers (inputs + out/lse/
    grads) and passes them in; the worker writes results in place. `run(...)` returns "ok" or
    "launch_fail" (config not implementable on this GPU/hub build), or raises on a worker error.
    """

    def __init__(self, device="cuda:0"):
        self.device = device
        ctx = mp.get_context("spawn")
        self._task_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_main, args=(self._task_q, self._res_q, device), daemon=True
        )
        self._proc.start()
        status, payload = self._res_q.get()
        if status != "ready":
            raise RuntimeError(f"Hub worker failed to start: {payload}")

    def run(self, task: dict) -> str:
        """Run one op (task['cmd']=='run'). Returns 'ok' or 'launch_fail'.

        Inputs must be globally complete before the worker (a different process/stream) reads
        them, so sync the producer side first.
        """
        task["cmd"] = "run"
        torch.cuda.synchronize()
        self._task_q.put(task)
        status, payload = self._res_q.get()
        if status == "error":
            raise RuntimeError(f"Hub worker error: {payload}")
        return status

    def get_configs(self, backend, q, k, v):
        """Ask the worker for the hub-advertised (fwd, bwd) configs for these inputs."""
        self._task_q.put({"cmd": "configs", "backend": backend, "q": q, "k": k, "v": v})
        status, payload = self._res_q.get()
        if status == "error":
            raise RuntimeError(f"Hub worker error: {payload}")
        return payload

    def close(self):
        if self._proc.is_alive():
            self._task_q.put(None)
            self._proc.join(timeout=15)
            if self._proc.is_alive():
                self._proc.terminate()
