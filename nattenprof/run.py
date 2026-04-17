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

"""CLI dispatch: parse args -> build problem -> dry-run/optimize/profile -> output."""

import torch
from natten.utils.checks import (
    check_all_args,
    check_input_size_arg,
    check_kernel_size_arg,
)

from nattenprof.engine import (
    DTYPE_MAP,
    profile_attn,
    profile_na,
    profile_sdpa,
    ProfileOptions,
)
from nattenprof.output import (
    build_output_json,
    get_metadata,
    print_profile_table,
    write_json,
)
from nattenprof.problem import AttentionProblem, NAProblem


def _setup_runtime(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)


# ---- Subcommand handlers ----


def _run_na(args):
    torch_dtype = DTYPE_MAP[args.dtype]
    na_dim = len(args.input_size)
    input_size = check_input_size_arg(na_dim, args.input_size)
    window_size = (
        check_kernel_size_arg(na_dim, args.window_size)
        if args.window_size
        else input_size
    )
    window_size, stride, dilation, causal = check_all_args(
        na_dim, window_size, args.stride, args.dilation, args.causal
    )

    problem = NAProblem(
        batch_size=args.batch_size,
        heads=args.heads,
        heads_kv=args.heads_kv,
        dim=args.dim,
        dim_value=args.dim_value,
        input_size=input_size,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        is_causal=causal,
        dtype=torch_dtype,
        additional_kv_length=args.add_kv,
    )

    # Dry-run
    if args.dry_run:
        from nattenprof.dry_run import dry_run_na

        dry_run_na(
            problem=problem,
            backend=args.backend,
            fmha_backend=args.fmha_backend,
            backprop=args.bwd,
            torch_compile=args.compile,
            max_configs=args.max_configs,
        )
        return None

    # Optimize
    if args.optimize:
        from nattenprof.dry_run import optimize_na

        best_cfg = optimize_na(
            problem=problem,
            backend=args.backend,
            fmha_backend=args.fmha_backend,
            backprop=args.bwd,
            persistent=args.persistent,
            schedule=args.schedule,
            torch_compile=args.compile,
            warmup_steps=args.optimize_warmup_steps,
            init_mode_str=args.init_mode,
            memory_limit=args.memory_limit,
            seed=args.seed,
        )
        args.backend = best_cfg.get("backend", args.backend)
        args.fmha_backend = best_cfg.get("fmha_backend", args.fmha_backend)
        args.q_tile = (
            list(best_cfg["q_tile_shape"])
            if "q_tile_shape" in best_cfg
            else args.q_tile
        )
        args.kv_tile = (
            list(best_cfg["kv_tile_shape"])
            if "kv_tile_shape" in best_cfg
            else args.kv_tile
        )
        args.backward_q_tile = (
            list(best_cfg["backward_q_tile_shape"])
            if "backward_q_tile_shape" in best_cfg
            else args.backward_q_tile
        )
        args.backward_kv_tile = (
            list(best_cfg["backward_kv_tile_shape"])
            if "backward_kv_tile_shape" in best_cfg
            else args.backward_kv_tile
        )
        if "kernel_schedule" in best_cfg:
            args.schedule = best_cfg["kernel_schedule"]

    opts = ProfileOptions(
        backend=args.backend,
        fmha_backend=args.fmha_backend,
        q_tile=tuple(args.q_tile) if args.q_tile else None,
        kv_tile=tuple(args.kv_tile) if args.kv_tile else None,
        bwd_q_tile=tuple(args.backward_q_tile) if args.backward_q_tile else None,
        bwd_kv_tile=tuple(args.backward_kv_tile) if args.backward_kv_tile else None,
        persistent=args.persistent,
        schedule=args.schedule,
        compile=args.compile,
        bwd=args.bwd,
        warmup_steps=args.warmup_steps,
        init_mode=args.init_mode,
        memory_limit=args.memory_limit,
        seed=args.seed,
    )
    return profile_na(problem, opts)


def _run_attn(args):
    torch_dtype = DTYPE_MAP[args.dtype]
    seqlen_kv = args.seqlen_kv if args.seqlen_kv is not None else args.seqlen

    seqlens_q = None
    seqlens_kv = None
    if args.varlen:
        if args.seqlens is None:
            raise ValueError("--seqlens required when --varlen is set.")
        seqlens_q = args.seqlens
        seqlens_kv = args.seqlens_kv

    problem = AttentionProblem(
        batch_size=args.batch_size,
        heads=args.heads,
        heads_kv=args.heads_kv,
        dim=args.dim,
        dim_value=args.dim_value,
        seqlen_q=args.seqlen,
        seqlen_kv=seqlen_kv,
        dtype=torch_dtype,
        is_causal=args.is_causal,
        seqlens_q=seqlens_q,
        seqlens_kv=seqlens_kv,
    )

    # Dry-run
    if args.dry_run:
        from nattenprof.dry_run import dry_run_attn

        dry_run_attn(
            problem=problem,
            backend=args.backend,
            backprop=args.bwd,
            torch_compile=args.compile,
            max_configs=args.max_configs,
        )
        return None

    # Optimize
    if args.optimize:
        from nattenprof.dry_run import optimize_attn

        best_cfg = optimize_attn(
            problem=problem,
            backend=args.backend,
            backprop=args.bwd,
            persistent=args.persistent,
            schedule=args.schedule,
            torch_compile=args.compile,
            warmup_steps=args.optimize_warmup_steps,
            init_mode_str=args.init_mode,
            memory_limit=args.memory_limit,
            seed=args.seed,
        )
        args.backend = best_cfg.get("backend", args.backend)
        args.q_tile = best_cfg.get("q_tile_size", args.q_tile)
        args.kv_tile = best_cfg.get("kv_tile_size", args.kv_tile)
        args.backward_q_tile = best_cfg.get(
            "backward_q_tile_size", args.backward_q_tile
        )
        args.backward_kv_tile = best_cfg.get(
            "backward_kv_tile_size", args.backward_kv_tile
        )
        if "kernel_schedule" in best_cfg:
            args.schedule = best_cfg["kernel_schedule"]

    opts = ProfileOptions(
        backend=args.backend,
        q_tile=args.q_tile,
        kv_tile=args.kv_tile,
        bwd_q_tile=args.backward_q_tile,
        bwd_kv_tile=args.backward_kv_tile,
        persistent=args.persistent,
        schedule=args.schedule,
        compile=args.compile,
        bwd=args.bwd,
        warmup_steps=args.warmup_steps,
        init_mode=args.init_mode,
        memory_limit=args.memory_limit,
        seed=args.seed,
    )
    return profile_attn(problem, opts)


def _run_sdpa(args):
    torch_dtype = DTYPE_MAP[args.dtype]
    seqlen_kv = args.seqlen_kv if args.seqlen_kv is not None else args.seqlen

    problem = AttentionProblem(
        batch_size=args.batch_size,
        heads=args.heads,
        heads_kv=args.heads_kv,
        dim=args.dim,
        dim_value=args.dim_value,
        seqlen_q=args.seqlen,
        seqlen_kv=seqlen_kv,
        dtype=torch_dtype,
        is_causal=args.is_causal,
    )

    opts = ProfileOptions(
        backend=args.backend,
        bwd=args.bwd,
        warmup_steps=args.warmup_steps,
        init_mode=args.init_mode,
        memory_limit=args.memory_limit,
        seed=args.seed,
    )
    return profile_sdpa(problem, opts)


def dispatch(args):
    _setup_runtime(args)

    if args.subcommand == "batch":
        from nattenprof.batch import run_batch

        run_batch(args)
        return

    if args.subcommand == "na":
        result = _run_na(args)
    elif args.subcommand == "attn":
        result = _run_attn(args)
    elif args.subcommand == "sdpa":
        result = _run_sdpa(args)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    if result is None:
        return

    print_profile_table(
        result.kernels, use_case_str=result.use_case_str, symbols=args.symbols
    )

    if args.output_json:
        metadata = get_metadata()
        data = build_output_json([result], metadata, symbols=args.symbols)
        write_json(data, args.output_json)
        print(f"Results written to {args.output_json}")
