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

"""Config enumeration (dry-run) and optimize search.

High-level entry points (`_dry_run_na`, `_dry_run_attn`, `_optimize_na`,
`_optimize_attn`) are called by Problem methods (NAProblem.dry_run /
AttnProblem.dry_run / etc.). All perf knobs come off the Problem.
"""

import math
from functools import partial
from typing import Any, Dict, List, Tuple

import torch

from nattenprof.output import print_table, progress_bar
from nattenprof.problem import AttnProblem, NAProblem

# ---- Temp tensor helpers ----


def _make_temp_tensors(problem):
    """Minimal temporary tensors for natten's backend compatibility helpers.

    Uses problem.get_tensor_shapes(), which already respects problem.heads_last.
    Short-lived; not part of the TensorPool.
    """
    shapes = problem.get_tensor_shapes()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_dtype = problem.dtype
    if safe_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        safe_dtype = torch.float16

    q = torch.randn(shapes["q"], dtype=safe_dtype, device=device)
    k = torch.randn(shapes["k"], dtype=safe_dtype, device=device)
    v = torch.randn(shapes["v"], dtype=safe_dtype, device=device)

    if problem.dtype != safe_dtype:
        q = q.to(problem.dtype)
        k = k.to(problem.dtype)
        v = v.to(problem.dtype)

    return q, k, v


# ---- Config gathering (no printing) ----


def _flatten_configs(configs, keys) -> List[Dict[str, Any]]:
    """Convert raw config tuples + keys into flat annotated dicts."""
    result = []
    for cfg in configs:
        assert len(cfg) == len(keys)
        annotated: Dict[str, Any] = {}
        for key, val in zip(keys, cfg):
            if isinstance(key, tuple):
                for sub_key, sub_val in zip(key, val):
                    annotated[sub_key] = sub_val
            else:
                annotated[key] = val
        result.append(annotated)
    return result


# Per-backend dispatch: natten.backends function names + config key schemas.
# FNA tiles are shape tuples ("*_tile_shape"); FMHA tiles are scalar sizes
# ("*_tile_size"). See _get_fna_configs vs _get_fmha_configs for the key diff.
_FNA_SPECS: Dict[str, Dict[str, Any]] = {
    "cutlass-fna": {
        "fwd_fn": "get_configs_for_cutlass_fna",
        "bwd_fn": "get_bwd_configs_for_cutlass_fna",
        "fwd_keys": ("q_tile_shape", "kv_tile_shape"),
        "bwd_keys": ("backward_q_tile_shape", "backward_kv_tile_shape"),
    },
    "hopper-fna": {
        "fwd_fn": "get_configs_for_cutlass_hopper_fna",
        "bwd_fn": "get_bwd_configs_for_cutlass_hopper_fna",
        "fwd_keys": (("q_tile_shape", "kv_tile_shape"), "kernel_schedule"),
        "bwd_keys": ("backward_q_tile_shape", "backward_kv_tile_shape"),
    },
    "blackwell-fna": {
        "fwd_fn": "get_configs_for_cutlass_blackwell_fna",
        "bwd_fn": "get_bwd_configs_for_cutlass_blackwell_fna",
        "fwd_keys": ("q_tile_shape", "kv_tile_shape"),
        "bwd_keys": ("backward_q_tile_shape", "backward_kv_tile_shape"),
    },
    "flex-fna": {
        "fwd_fn": "get_configs_for_flex_fna",
        "bwd_fn": None,
        "fwd_keys": ("q_tile_shape", "kv_tile_shape"),
        "bwd_keys": (),
    },
}

_FMHA_SPECS: Dict[str, Dict[str, Any]] = {
    "cutlass-fmha": {
        "fwd_fn": "get_configs_for_cutlass_fmha",
        "bwd_fn": "get_bwd_configs_for_cutlass_fmha",
        "fwd_keys": ("q_tile_size", "kv_tile_size"),
        "bwd_keys": ("backward_q_tile_size", "backward_kv_tile_size"),
    },
    "hopper-fmha": {
        "fwd_fn": "get_configs_for_cutlass_hopper_fmha",
        "bwd_fn": "get_bwd_configs_for_cutlass_hopper_fmha",
        "fwd_keys": (("q_tile_size", "kv_tile_size"), "kernel_schedule"),
        "bwd_keys": ("backward_q_tile_size", "backward_kv_tile_size"),
    },
    "blackwell-fmha": {
        "fwd_fn": "get_configs_for_cutlass_blackwell_fmha",
        "bwd_fn": "get_bwd_configs_for_cutlass_blackwell_fmha",
        "fwd_keys": ("q_tile_size", "kv_tile_size"),
        "bwd_keys": ("backward_q_tile_size", "backward_kv_tile_size"),
    },
    "flex-fmha": {
        "fwd_fn": "get_configs_for_flex_fmha",
        "bwd_fn": None,
        "fwd_keys": ("q_tile_size", "kv_tile_size"),
        "bwd_keys": (),
    },
}


# Keys used to carry tiles through optimize search + apply. These match both
# natten's config-dict keys (what the search returns) and PerfConfig field names.
_FNA_FWD_KEYS = ("q_tile_shape", "kv_tile_shape")
_FNA_BWD_KEYS = ("backward_q_tile_shape", "backward_kv_tile_shape")
_FMHA_FWD_KEYS = ("q_tile_size", "kv_tile_size")
_FMHA_BWD_KEYS = ("backward_q_tile_size", "backward_kv_tile_size")


def _get_configs(
    specs: Dict[str, Dict[str, Any]], family: str, backend: str, q, k, v
) -> Tuple[List[Dict], List[Dict]]:
    import natten.backends as nb

    if backend not in specs:
        raise ValueError(f"Unsupported {family} backend: {backend}")
    spec = specs[backend]

    fwd_raw = getattr(nb, spec["fwd_fn"])(q, k, v)
    bwd_raw = getattr(nb, spec["bwd_fn"])(q, k, v) if spec["bwd_fn"] else []

    fwd = _flatten_configs(fwd_raw, spec["fwd_keys"]) if fwd_raw else []
    bwd = _flatten_configs(bwd_raw, spec["bwd_keys"]) if bwd_raw else []
    return fwd, bwd


def _get_fna_configs(backend: str, q, k, v) -> Tuple[List[Dict], List[Dict]]:
    fwd, bwd = _get_configs(_FNA_SPECS, "FNA", backend, q, k, v)
    # FNA emits shape tuples; derive scalar tile_size for downstream consumers.
    for cfg in fwd:
        if "q_tile_shape" in cfg:
            cfg["q_tile_size"] = math.prod(cfg["q_tile_shape"])
        if "kv_tile_shape" in cfg:
            cfg["kv_tile_size"] = math.prod(cfg["kv_tile_shape"])
    for cfg in bwd:
        if "backward_q_tile_shape" in cfg:
            cfg["backward_q_tile_size"] = math.prod(cfg["backward_q_tile_shape"])
        if "backward_kv_tile_shape" in cfg:
            cfg["backward_kv_tile_size"] = math.prod(cfg["backward_kv_tile_shape"])
    return fwd, bwd


def _get_fmha_configs(backend: str, q, k, v) -> Tuple[List[Dict], List[Dict]]:
    return _get_configs(_FMHA_SPECS, "FMHA", backend, q, k, v)


# ---- Display ----


def _display_configs(title, configs, max_configs):
    if not configs:
        return
    headers = list(configs[0].keys())
    values = [[str(v) for v in cfg.values()] for cfg in configs]
    if max_configs > 0 and len(values) > max_configs:
        values = values[:max_configs]
        values.append(["..." for _ in headers])
    print_table(title, headers, values, has_footer=False)


# ---- Dry-run impls (called from Problem.dry_run) ----


def _dry_run_na(problem: NAProblem, max_configs: int) -> None:
    from natten.backends import get_compatible_backends, get_compatible_fmha_backends

    q, k, v = _make_temp_tensors(problem)
    perf = problem.perf

    if problem.is_self_attn():
        q_flat = q.flatten(1, problem.na_dim)
        k_flat = k.flatten(1, problem.na_dim)
        v_flat = v.flatten(1, problem.na_dim)

        fmha_backends = (
            [perf.fmha_backend]
            if perf.fmha_backend is not None
            else get_compatible_fmha_backends(
                q_flat,
                k_flat,
                v_flat,
                torch_compile=perf.torch_compile,
                is_causal=False,
                is_varlen=False,
            )
        )
        for fb in fmha_backends:
            print(f"Use case is compatible with backend {fb}.")
            fwd, bwd = _get_fmha_configs(fb, q_flat, k_flat, v_flat)
            if fwd:
                _display_configs(
                    f"Backend: {fb}\nForward pass configurations", fwd, max_configs
                )
            if problem.bwd and bwd:
                _display_configs(
                    f"Backend: {fb}\nBackward pass configurations", bwd, max_configs
                )
        return

    backends = (
        [perf.fna_backend]
        if perf.fna_backend is not None
        else get_compatible_backends(q, k, v, torch_compile=perf.torch_compile)
    )
    for b in backends:
        print(f"Use case is compatible with backend {b}.")
        fwd, bwd = _get_fna_configs(b, q, k, v)
        if fwd:
            _display_configs(
                f"Backend: {b}\nForward pass configurations", fwd, max_configs
            )
        if problem.bwd and bwd:
            _display_configs(
                f"Backend: {b}\nBackward pass configurations", bwd, max_configs
            )


def _dry_run_attn(problem: AttnProblem, max_configs: int) -> None:
    from natten.backends import get_compatible_fmha_backends

    q, k, v = _make_temp_tensors(problem)
    perf = problem.perf

    backends = (
        [perf.fmha_backend]
        if perf.fmha_backend is not None
        else get_compatible_fmha_backends(
            q,
            k,
            v,
            torch_compile=perf.torch_compile,
            is_causal=problem.is_causal,
            is_varlen=problem.is_varlen,
        )
    )
    for b in backends:
        print(f"Use case is compatible with backend {b}.")
        fwd, bwd = _get_fmha_configs(b, q, k, v)
        if fwd:
            _display_configs(
                f"Backend: {b}\nForward pass configurations", fwd, max_configs
            )
        if problem.bwd and bwd:
            _display_configs(
                f"Backend: {b}\nBackward pass configurations", bwd, max_configs
            )


# ---- Optimize impls (called from Problem.optimize) ----


def _run_optimize_loop(configs, measure_fn, warmup_steps):
    best_time = None
    best_config = None
    best_time_str = None
    for cfg in progress_bar(configs, total=len(configs)):
        runtime_ms = measure_fn(cfg, warmup_steps)
        runtime_str = f"{runtime_ms:.2f} ms"
        if best_time is None or runtime_ms < best_time:
            best_time = runtime_ms
            best_config = cfg
            best_time_str = runtime_str
    assert best_config is not None and best_time_str is not None
    return best_config, best_time_str


def _make_pool(problem, settings, requires_grad: bool):
    from nattenprof.tensors import InitMode, TensorPool

    return TensorPool(
        shapes=problem.get_tensor_shapes(),
        dtype=problem.dtype,
        device=torch.device("cuda"),
        init_mode=InitMode(settings.init_mode),
        memory_limit_gb=settings.memory_limit,
        seed=settings.seed,
        requires_grad=requires_grad,
    )


def _run_search_apply(
    problem, settings, fwd_configs, bwd_configs, make_measure, fwd_keys, bwd_keys
) -> None:
    """Run fwd/optional bwd search loops and apply best-of to problem.perf.

    `fwd_keys` / `bwd_keys` are tuples of perf field names that also appear as
    keys in the cfg dicts (matching natten's parameter names). The assignment is
    a silent no-op if a given key isn't in the best cfg.
    """
    perf = problem.perf

    print()
    print(f"Searching {len(fwd_configs)} forward pass configs")
    best_fwd, _ = _run_optimize_loop(
        fwd_configs, make_measure(run_backprop=False), settings.warmup_steps
    )
    for k in fwd_keys:
        if k in best_fwd:
            setattr(perf, k, best_fwd[k])
    if "kernel_schedule" in best_fwd:
        perf.kernel_schedule = best_fwd["kernel_schedule"]

    if problem.bwd and bwd_configs:
        print()
        print(f"Searching {len(bwd_configs)} backward pass configs")
        best_bwd, _ = _run_optimize_loop(
            bwd_configs, make_measure(run_backprop=True), settings.warmup_steps
        )
        for k in bwd_keys:
            if k in best_bwd:
                setattr(perf, k, best_bwd[k])

    _print_best(problem)


def _optimize_na(problem: NAProblem, settings) -> None:
    """Search configs; mutate problem's backend / tiles / schedule in place."""
    from natten import set_memory_usage_preference, use_kv_parallelism_in_fused_na
    from natten.backends import choose_backend, choose_fmha_backend

    from nattenprof.engine import measure_wall_time_ms
    from nattenprof.ops import run_na

    use_kv_parallelism_in_fused_na(True)
    set_memory_usage_preference("unrestricted")

    q, k, v = _make_temp_tensors(problem)
    perf = problem.perf

    # Pick a concrete backend to drive the search (FNA or FMHA via self-attn path).
    if problem.is_self_attn():
        q_flat = q.flatten(1, problem.na_dim)
        k_flat = k.flatten(1, problem.na_dim)
        v_flat = v.flatten(1, problem.na_dim)
        perf.fmha_backend = perf.fmha_backend or choose_fmha_backend(
            q_flat,
            k_flat,
            v_flat,
            torch_compile=perf.torch_compile,
            is_causal=False,
            is_varlen=False,
        )
        fwd_configs, bwd_configs = _get_fmha_configs(
            perf.fmha_backend, q_flat, k_flat, v_flat
        )
        fwd_keys, bwd_keys = _FMHA_FWD_KEYS, _FMHA_BWD_KEYS
    else:
        perf.fna_backend = perf.fna_backend or choose_backend(
            q, k, v, torch_compile=perf.torch_compile
        )
        fwd_configs, bwd_configs = _get_fna_configs(perf.fna_backend, q, k, v)
        fwd_keys, bwd_keys = _FNA_FWD_KEYS, _FNA_BWD_KEYS

    torch.set_grad_enabled(problem.bwd)

    def make_measure(run_backprop: bool):
        pool = _make_pool(problem, settings, requires_grad=run_backprop)

        def measure(cfg: Dict, warmup: int) -> float:
            fn = partial(
                run_na,
                problem=problem,
                backend=perf.fna_backend,
                fmha_backend=perf.fmha_backend,
                q_tile_shape=cfg.get("q_tile_shape"),
                kv_tile_shape=cfg.get("kv_tile_shape"),
                backward_q_tile_shape=cfg.get("backward_q_tile_shape"),
                backward_kv_tile_shape=cfg.get("backward_kv_tile_shape"),
                q_tile_size=cfg.get("q_tile_size"),
                kv_tile_size=cfg.get("kv_tile_size"),
                backward_q_tile_size=cfg.get("backward_q_tile_size"),
                backward_kv_tile_size=cfg.get("backward_kv_tile_size"),
                run_persistent_kernel=perf.is_persistent,
                kernel_schedule=cfg.get("kernel_schedule", perf.kernel_schedule),
                torch_compile=perf.torch_compile,
                disable_backward=not run_backprop,
            )
            return measure_wall_time_ms(pool, fn, warmup_steps=warmup)

        return measure

    _run_search_apply(
        problem, settings, fwd_configs, bwd_configs, make_measure, fwd_keys, bwd_keys
    )


def _optimize_attn(problem: AttnProblem, settings) -> None:
    """Search FMHA configs; mutate problem in place."""
    from natten.backends import choose_fmha_backend

    from nattenprof.engine import measure_wall_time_ms
    from nattenprof.ops import run_attn

    q, k, v = _make_temp_tensors(problem)
    perf = problem.perf
    perf.fmha_backend = perf.fmha_backend or choose_fmha_backend(
        q,
        k,
        v,
        torch_compile=perf.torch_compile,
        is_causal=problem.is_causal,
        is_varlen=problem.is_varlen,
    )
    fwd_configs, bwd_configs = _get_fmha_configs(perf.fmha_backend, q, k, v)

    torch.set_grad_enabled(problem.bwd)

    def make_measure(run_backprop: bool):
        pool = _make_pool(problem, settings, requires_grad=run_backprop)

        def measure(cfg: Dict, warmup: int) -> float:
            fn = partial(
                run_attn,
                problem=problem,
                backend=perf.fmha_backend,
                q_tile_size=cfg.get("q_tile_size"),
                kv_tile_size=cfg.get("kv_tile_size"),
                backward_q_tile_size=cfg.get("backward_q_tile_size"),
                backward_kv_tile_size=cfg.get("backward_kv_tile_size"),
                run_persistent_kernel=perf.is_persistent,
                kernel_schedule=cfg.get("kernel_schedule", perf.kernel_schedule),
                torch_compile=perf.torch_compile,
                disable_backward=not run_backprop,
            )
            return measure_wall_time_ms(pool, fn, warmup_steps=warmup)

        return measure

    _run_search_apply(
        problem,
        settings,
        fwd_configs,
        bwd_configs,
        make_measure,
        _FMHA_FWD_KEYS,
        _FMHA_BWD_KEYS,
    )


def _print_best(problem) -> None:
    perf = problem.perf
    print()
    print_table(
        "Best configuration",
        ["Parameter", "Value"],
        [
            ["fna_backend", str(perf.fna_backend)],
            ["fmha_backend", str(perf.fmha_backend)],
            ["q_tile_shape", str(perf.q_tile_shape)],
            ["kv_tile_shape", str(perf.kv_tile_shape)],
            ["backward_q_tile_shape", str(perf.backward_q_tile_shape)],
            ["backward_kv_tile_shape", str(perf.backward_kv_tile_shape)],
            ["q_tile_size", str(perf.q_tile_size)],
            ["kv_tile_size", str(perf.kv_tile_size)],
            ["backward_q_tile_size", str(perf.backward_q_tile_size)],
            ["backward_kv_tile_size", str(perf.backward_kv_tile_size)],
            ["kernel_schedule", str(perf.kernel_schedule)],
        ],
        has_footer=False,
    )
    print()
