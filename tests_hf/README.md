# tests_hf — HuggingFace-hub NATTEN parity tests

Parity tests between the **HF-hub packaged NATTEN** (`kernels-staging/natten`) and the **locally
installed `natten`**. Structure mirrors `tests/`: **one file per backend** (backends are different
kernels), not one file per op — each backend file exercises na1d/na2d/na3d (or `attention` for
FMHA) through that backend.

Flex backends are intentionally out of scope.

## What it checks

For each backend, for a set of curated + random problem shapes, sweep dtypes and a few of the
**hub-advertised** tile configs, and compare hub vs local on identical inputs:

| Tensor        | Tolerance                                                              |
|---------------|-----------------------------------------------------------------------|
| forward / LSE | bit-exact (`atol=rtol=0`)                                             |
| dK, dV        | bit-exact (`atol=rtol=0`)                                             |
| dQ            | bit-exact **iff** deterministic mode AND backend ∈ {cutlass-fna, cutlass-fmha}; otherwise `atol=1e-2` |

dQ uses an atomic (kv-parallel) reduction by default, which is not reproducible run-to-run — so it
is only bit-exact under NATTEN's deterministic backward, and only for the cutlass backends. The
relaxed tolerance is `1e-2` (`utils.DQ_RELAXED_ATOL`); do not loosen past `1e-1`.

## Why a separate process (the important part)

Two `libnatten.so` **cannot coexist in one process**. `RTLD_DEEPBIND` isolates *host* symbols, and
that is enough for the cutlass-2.x kernels — but not for the sm100 (Blackwell) CUTLASS-3.x kernels.
Those go through process-global CUDA runtime state: CUDA APIs like
`cudaFuncSetAttribute(kernel, MaxDynamicSharedMemorySize, ...)` key off the host kernel-function
stub, and with two fatbins registering identically-named `cutlass::device_kernel_sm100<...>`
kernels the driver routes the launch to the wrong module → the dynamic-smem opt-in is lost →
`Failed to launch the CUTLASS kernel` and NaN. Host-symbol isolation cannot fix that.

So the hub kernel runs in a **persistent child process** (`hub_process.py`): hub loaded there,
local `natten` here, neither co-loads. The child is spawned **once** and reused for the whole
session. Parent and child share GPU tensors via **CUDA IPC** (`torch.multiprocessing` spawn): the
parent allocates every buffer (inputs + out/lse/grads) and the worker writes results in place, so
the parent owns all memory.

Cross-process ordering: the producer synchronizes at each hand-off — the parent before the worker
reads inputs, the worker after it writes outputs — since the two processes use different CUDA
streams and a queue round-trip only gives CPU ordering, not device completion.

`hub_process.py` must **never import `natten`** (it is re-imported in the spawned child; importing
natten there would recreate the co-load collision).

## Running

```bash
.venv/bin/python -m pytest tests_hf/ -q
```

First run downloads + caches the hub wheel (`~/.cache/huggingface/`). Needs a CUDA GPU.

## Honest coverage accounting

A config forced from the hub may not be launchable on the current GPU/build. libnatten prints
`Failed to launch the CUTLASS kernel` and returns garbage instead of raising, so the worker
captures fd 2, detects that, and reports `launch_fail` → the parent **skips** the case (counted,
never silently dropped). A backend test whose cases were *all* skipped reports as **SKIPPED**, not
passed. The terminal summary prints the number of real bit-exact comparisons per backend plus skip
counts.

Random problem generators are **per backend** (`_FNA_RANDSWEEP` / `_FMHA_RANDSWEEP` in `utils.py`),
mirroring the corresponding generators in `tests/`: each backend supports a different set of
head_dims, whether `head_dim_v != head_dim` is allowed, GQA, and 3D size caps. A single generic
generator would emit shapes a backend can't run (empty config list → skips), so each is
constrained to its backend's supported set.

## Known environment notes (B200 / SM100)

- **cutlass-fna, cutlass-fmha, blackwell-fna, blackwell-fmha**: all fully exercised, bit-exact,
  nothing skipped.
- **hopper-fna, hopper-fmha**: skipped — require SM90 hardware (not present on B200).
