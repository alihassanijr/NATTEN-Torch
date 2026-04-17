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

This module creates temporary tensors for backend compatibility checks and config
enumeration. These are short-lived and not part of the TensorPool system.
"""

import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from nattenprof.output import print_table, progress_bar
from nattenprof.problem import AttentionProblem, NAProblem

# ---- Temp tensor helpers ----


def _make_temp_tensors(problem, heads_last=True):
    """Create minimal temporary tensors for config enumeration."""
    shapes = (
        problem.get_tensor_shapes(heads_last=heads_last)
        if isinstance(problem, AttentionProblem)
        else problem.get_tensor_shapes()
    )

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


def _get_fna_configs(backend: str, q, k, v) -> Tuple[List[Dict], List[Dict]]:
    """Get flattened fwd and bwd configs for an FNA backend."""
    from natten.backends import (
        get_bwd_configs_for_cutlass_blackwell_fna,
        get_bwd_configs_for_cutlass_fna,
        get_bwd_configs_for_cutlass_hopper_fna,
        get_configs_for_cutlass_blackwell_fna,
        get_configs_for_cutlass_fna,
        get_configs_for_cutlass_hopper_fna,
        get_configs_for_flex_fna,
    )

    if backend == "cutlass-fna":
        fwd_raw = get_configs_for_cutlass_fna(q, k, v)
        bwd_raw = get_bwd_configs_for_cutlass_fna(q, k, v)
        fwd_keys = ("q_tile_shape", "kv_tile_shape")
        bwd_keys = ("backward_q_tile_shape", "backward_kv_tile_shape")

    elif backend == "hopper-fna":
        fwd_raw = get_configs_for_cutlass_hopper_fna(q, k, v)
        bwd_raw = get_bwd_configs_for_cutlass_hopper_fna(q, k, v)
        fwd_keys = (("q_tile_shape", "kv_tile_shape"), "kernel_schedule")
        bwd_keys = ("backward_q_tile_shape", "backward_kv_tile_shape")

    elif backend == "blackwell-fna":
        fwd_raw = get_configs_for_cutlass_blackwell_fna(q, k, v)
        bwd_raw = get_bwd_configs_for_cutlass_blackwell_fna(q, k, v)
        fwd_keys = ("q_tile_shape", "kv_tile_shape")
        bwd_keys = ("backward_q_tile_shape", "backward_kv_tile_shape")

    elif backend == "flex-fna":
        fwd_raw = get_configs_for_flex_fna(q, k, v)
        bwd_raw = []
        fwd_keys = ("q_tile_shape", "kv_tile_shape")
        bwd_keys = ()

    else:
        raise ValueError(f"Unsupported FNA backend: {backend}")

    fwd = _flatten_configs(fwd_raw, fwd_keys) if fwd_raw else []
    bwd = _flatten_configs(bwd_raw, bwd_keys) if bwd_raw else []

    # Add q_tile_size / kv_tile_size for compatibility with measure functions
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
    """Get flattened fwd and bwd configs for an FMHA backend."""
    from natten.backends import (
        get_bwd_configs_for_cutlass_blackwell_fmha,
        get_bwd_configs_for_cutlass_fmha,
        get_bwd_configs_for_cutlass_hopper_fmha,
        get_configs_for_cutlass_blackwell_fmha,
        get_configs_for_cutlass_fmha,
        get_configs_for_cutlass_hopper_fmha,
        get_configs_for_flex_fmha,
    )

    if backend == "cutlass-fmha":
        fwd_raw = get_configs_for_cutlass_fmha(q, k, v)
        bwd_raw = get_bwd_configs_for_cutlass_fmha(q, k, v)
        fwd_keys = ("q_tile_size", "kv_tile_size")
        bwd_keys = ("backward_q_tile_size", "backward_kv_tile_size")

    elif backend == "hopper-fmha":
        fwd_raw = get_configs_for_cutlass_hopper_fmha(q, k, v)
        bwd_raw = get_bwd_configs_for_cutlass_hopper_fmha(q, k, v)
        fwd_keys = (("q_tile_size", "kv_tile_size"), "kernel_schedule")
        bwd_keys = ("backward_q_tile_shape", "backward_kv_tile_shape")

    elif backend == "blackwell-fmha":
        fwd_raw = get_configs_for_cutlass_blackwell_fmha(q, k, v)
        bwd_raw = get_bwd_configs_for_cutlass_blackwell_fmha(q, k, v)
        fwd_keys = ("q_tile_size", "kv_tile_size")
        bwd_keys = ("backward_q_tile_size", "backward_kv_tile_size")

    elif backend == "flex-fmha":
        fwd_raw = get_configs_for_flex_fmha(q, k, v)
        bwd_raw = []
        fwd_keys = ("q_tile_size", "kv_tile_size")
        bwd_keys = ()

    else:
        raise ValueError(f"Unsupported FMHA backend: {backend}")

    fwd = _flatten_configs(fwd_raw, fwd_keys) if fwd_raw else []
    bwd = _flatten_configs(bwd_raw, bwd_keys) if bwd_raw else []
    return fwd, bwd


# ---- Display helpers ----


def _display_configs(title, configs, max_configs):
    """Print config dicts as a table."""
    if not configs:
        return

    headers = list(configs[0].keys())
    values = [[str(v) for v in cfg.values()] for cfg in configs]

    if max_configs > 0 and len(values) > max_configs:
        values = values[:max_configs]
        values.append(["..." for _ in headers])

    print_table(title, headers, values, has_footer=False)


# ---- Dry-run (NA) ----


def dry_run_na(
    problem: NAProblem,
    backend: Optional[str],
    fmha_backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
    max_configs: int,
):
    """Display available configs for NA backends."""
    from natten.backends import get_compatible_backends, get_compatible_fmha_backends

    q, k, v = _make_temp_tensors(problem)

    if problem.is_self_attn:
        q_flat = q.flatten(1, problem.na_dim)
        k_flat = k.flatten(1, problem.na_dim)
        v_flat = v.flatten(1, problem.na_dim)

        if fmha_backend is None:
            fmha_backends = get_compatible_fmha_backends(
                q_flat,
                k_flat,
                v_flat,
                torch_compile=torch_compile,
                is_causal=False,
                is_varlen=False,
            )
        else:
            fmha_backends = [fmha_backend]

        for fb in fmha_backends:
            print(f"Use case is compatible with backend {fb}.")
            fwd, bwd = _get_fmha_configs(fb, q_flat, k_flat, v_flat)
            if fwd:
                _display_configs(
                    f"Backend: {fb}\nForward pass configurations",
                    fwd,
                    max_configs,
                )
            if backprop and bwd:
                _display_configs(
                    f"Backend: {fb}\nBackward pass configurations",
                    bwd,
                    max_configs,
                )
    else:
        if backend is None:
            backends = get_compatible_backends(q, k, v, torch_compile=torch_compile)
        else:
            backends = [backend]

        for b in backends:
            print(f"Use case is compatible with backend {b}.")
            fwd, bwd = _get_fna_configs(b, q, k, v)
            if fwd:
                _display_configs(
                    f"Backend: {b}\nForward pass configurations",
                    fwd,
                    max_configs,
                )
            if backprop and bwd:
                _display_configs(
                    f"Backend: {b}\nBackward pass configurations",
                    bwd,
                    max_configs,
                )


# ---- Dry-run (attn) ----


def dry_run_attn(
    problem: AttentionProblem,
    backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
    max_configs: int,
):
    """Display available configs for FMHA backends."""
    from natten.backends import get_compatible_fmha_backends

    q, k, v = _make_temp_tensors(problem, heads_last=True)

    if backend is None:
        backends = get_compatible_fmha_backends(
            q,
            k,
            v,
            torch_compile=torch_compile,
            is_causal=problem.is_causal,
            is_varlen=problem.is_varlen,
        )
    else:
        backends = [backend]

    for b in backends:
        print(f"Use case is compatible with backend {b}.")
        fwd, bwd = _get_fmha_configs(b, q, k, v)
        if fwd:
            _display_configs(
                f"Backend: {b}\nForward pass configurations",
                fwd,
                max_configs,
            )
        if backprop and bwd:
            _display_configs(
                f"Backend: {b}\nBackward pass configurations",
                bwd,
                max_configs,
            )


# ---- Optimize ----


def _find_na_configs(
    problem: NAProblem,
    backend: Optional[str],
    fmha_backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
) -> Tuple[str, Optional[str], List[Dict], List[Dict]]:
    """Find backend + all configs for an NA problem. Returns (backend, fmha_backend, fwd, bwd)."""
    from natten.backends import choose_backend, choose_fmha_backend

    q, k, v = _make_temp_tensors(problem)

    if problem.is_self_attn:
        q_flat = q.flatten(1, problem.na_dim)
        k_flat = k.flatten(1, problem.na_dim)
        v_flat = v.flatten(1, problem.na_dim)
        resolved_fmha = fmha_backend or choose_fmha_backend(
            q_flat,
            k_flat,
            v_flat,
            torch_compile=torch_compile,
            is_causal=False,
            is_varlen=False,
        )
        fwd, bwd = _get_fmha_configs(resolved_fmha, q_flat, k_flat, v_flat)
        resolved_backend = backend or choose_backend(
            q, k, v, torch_compile=torch_compile
        )
        return resolved_backend, resolved_fmha, fwd, bwd
    else:
        resolved_backend = backend or choose_backend(
            q, k, v, torch_compile=torch_compile
        )
        fwd, bwd = _get_fna_configs(resolved_backend, q, k, v)
        resolved_fmha = fmha_backend
        return resolved_backend, resolved_fmha, fwd, bwd


def _find_attn_configs(
    problem: AttentionProblem,
    backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
) -> Tuple[str, List[Dict], List[Dict]]:
    """Find backend + all configs for an attention problem."""
    from natten.backends import choose_fmha_backend

    q, k, v = _make_temp_tensors(problem, heads_last=True)
    resolved = backend or choose_fmha_backend(
        q,
        k,
        v,
        torch_compile=torch_compile,
        is_causal=problem.is_causal,
        is_varlen=problem.is_varlen,
    )
    fwd, bwd = _get_fmha_configs(resolved, q, k, v)
    return resolved, fwd, bwd


def _run_optimize_loop(
    configs: List[Dict],
    measure_fn,
    warmup_steps: int,
) -> Tuple[Dict, str]:
    """Search configs, return (best_config, best_time_str)."""
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

    assert best_config is not None
    assert best_time_str is not None
    return best_config, best_time_str


def optimize_na(
    problem: NAProblem,
    backend: Optional[str],
    fmha_backend: Optional[str],
    backprop: bool,
    persistent: bool,
    schedule: Optional[str],
    torch_compile: bool,
    warmup_steps: int,
    init_mode_str: str,
    memory_limit: float,
    seed: int,
) -> Dict[str, Any]:
    """Run optimize for NA. Returns best config dict."""
    from natten import set_memory_usage_preference, use_kv_parallelism_in_fused_na

    use_kv_parallelism_in_fused_na(True)
    set_memory_usage_preference("unrestricted")

    resolved_backend, resolved_fmha, fwd_configs, bwd_configs = _find_na_configs(
        problem,
        backend=backend,
        fmha_backend=fmha_backend,
        backprop=backprop,
        torch_compile=torch_compile,
    )

    from nattenprof.engine import measure_wall_time_ms
    from nattenprof.ops import run_na
    from nattenprof.tensors import InitMode, TensorPool

    def make_measure_fn(run_backprop: bool):
        disable_backward = not run_backprop
        torch.set_grad_enabled(not disable_backward)

        pool = TensorPool(
            shapes=problem.get_tensor_shapes(),
            dtype=problem.dtype,
            device=torch.device("cuda"),
            init_mode=InitMode(init_mode_str),
            memory_limit_gb=memory_limit,
            seed=seed,
            requires_grad=not disable_backward,
        )

        def measure(cfg: Dict, warmup: int) -> float:
            fn = partial(
                run_na,
                problem=problem,
                backend=resolved_backend,
                fmha_backend=resolved_fmha,
                q_tile_shape=cfg.get("q_tile_shape"),
                kv_tile_shape=cfg.get("kv_tile_shape"),
                backward_q_tile_shape=cfg.get("backward_q_tile_shape"),
                backward_kv_tile_shape=cfg.get("backward_kv_tile_shape"),
                run_persistent_kernel=persistent,
                kernel_schedule=cfg.get("kernel_schedule", schedule),
                torch_compile=torch_compile,
                disable_backward=disable_backward,
            )
            return measure_wall_time_ms(pool, fn, warmup_steps=warmup)

        return measure

    best_cfg: Dict[str, Any] = {
        "backend": resolved_backend,
        "fmha_backend": resolved_fmha,
    }

    print()
    print(f"Searching {len(fwd_configs)} forward pass configs")
    fwd_measure = make_measure_fn(run_backprop=False)
    best_fwd, best_fwd_time = _run_optimize_loop(fwd_configs, fwd_measure, warmup_steps)
    best_cfg.update(best_fwd)

    if backprop and bwd_configs:
        print()
        print(f"Searching {len(bwd_configs)} backward pass configs")
        bwd_measure = make_measure_fn(run_backprop=True)
        best_bwd, best_bwd_time = _run_optimize_loop(
            bwd_configs, bwd_measure, warmup_steps
        )
        best_cfg.update(best_bwd)

    print()
    print_table(
        "Best configuration",
        ["Parameter", "Value"],
        [[k, str(v)] for k, v in best_cfg.items()],
        has_footer=False,
    )
    print()

    return best_cfg


def optimize_attn(
    problem: AttentionProblem,
    backend: Optional[str],
    backprop: bool,
    persistent: bool,
    schedule: Optional[str],
    torch_compile: bool,
    warmup_steps: int,
    init_mode_str: str,
    memory_limit: float,
    seed: int,
) -> Dict[str, Any]:
    """Run optimize for attention. Returns best config dict."""
    resolved_backend, fwd_configs, bwd_configs = _find_attn_configs(
        problem,
        backend=backend,
        backprop=backprop,
        torch_compile=torch_compile,
    )

    from nattenprof.engine import measure_wall_time_ms
    from nattenprof.ops import run_attn
    from nattenprof.tensors import InitMode, TensorPool

    def make_measure_fn(run_backprop: bool):
        disable_backward = not run_backprop
        torch.set_grad_enabled(not disable_backward)

        pool = TensorPool(
            shapes=problem.get_tensor_shapes(heads_last=True),
            dtype=problem.dtype,
            device=torch.device("cuda"),
            init_mode=InitMode(init_mode_str),
            memory_limit_gb=memory_limit,
            seed=seed,
            requires_grad=not disable_backward,
        )

        def measure(cfg: Dict, warmup: int) -> float:
            fn = partial(
                run_attn,
                problem=problem,
                backend=resolved_backend,
                q_tile_size=cfg.get("q_tile_size"),
                kv_tile_size=cfg.get("kv_tile_size"),
                backward_q_tile_size=cfg.get("backward_q_tile_size"),
                backward_kv_tile_size=cfg.get("backward_kv_tile_size"),
                run_persistent_kernel=persistent,
                kernel_schedule=cfg.get("kernel_schedule", schedule),
                torch_compile=torch_compile,
                disable_backward=disable_backward,
            )
            return measure_wall_time_ms(pool, fn, warmup_steps=warmup)

        return measure

    best_cfg: Dict[str, Any] = {
        "backend": resolved_backend,
    }

    print()
    print(f"Searching {len(fwd_configs)} forward pass configs")
    fwd_measure = make_measure_fn(run_backprop=False)
    best_fwd, best_fwd_time = _run_optimize_loop(fwd_configs, fwd_measure, warmup_steps)
    best_cfg.update(best_fwd)

    if backprop and bwd_configs:
        print()
        print(f"Searching {len(bwd_configs)} backward pass configs")
        bwd_measure = make_measure_fn(run_backprop=True)
        best_bwd, best_bwd_time = _run_optimize_loop(
            bwd_configs, bwd_measure, warmup_steps
        )
        best_cfg.update(best_bwd)

    print()
    print_table(
        "Best configuration",
        ["Parameter", "Value"],
        [[k, str(v)] for k, v in best_cfg.items()],
        has_footer=False,
    )
    print()

    return best_cfg
