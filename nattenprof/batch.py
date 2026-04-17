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

"""Batch mode: read JSON config file, run all entries, write JSON results."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List

from nattenprof.output import (
    build_output_json,
    get_metadata,
    print_profile_table,
    ProfileResult,
    write_json,
)
from nattenprof.run import _run_attn, _run_na, _run_sdpa


def _entry_to_args(entry: Dict[str, Any], runtime_args) -> SimpleNamespace:
    """Convert a batch JSON entry dict into a namespace that looks like CLI args.

    Merges entry-level params with runtime-level params from the CLI (init_mode,
    memory_limit, seed, device, etc.).
    """
    return SimpleNamespace(
        # Problem shape
        batch_size=entry.get("batch_size", 1),
        heads=entry.get("heads", 1),
        heads_kv=entry.get("heads_kv", entry.get("heads", 1)),
        dim=entry.get("dim", 64),
        dim_value=entry.get("dim_value", entry.get("dim", 64)),
        dtype=entry.get("dtype", "fp16"),
        # NA params
        input_size=entry.get("input_size", None),
        window_size=entry.get("window_size", None),
        stride=entry.get("stride", None),
        dilation=entry.get("dilation", None),
        causal=entry.get("is_causal", None),
        add_kv=entry.get("add_kv", 0),
        # Attn/SDPA params
        seqlen=entry.get("seqlen", None),
        seqlen_kv=entry.get("seqlen_kv", None),
        is_causal=entry.get("is_causal", False),
        varlen=entry.get("varlen", False),
        seqlens=entry.get("seqlens", None),
        seqlens_kv=entry.get("seqlens_kv", None),
        # Backend / perf
        backend=entry.get("backend", None),
        fmha_backend=entry.get("fmha_backend", None),
        q_tile=entry.get("q_tile", None),
        kv_tile=entry.get("kv_tile", None),
        backward_q_tile=entry.get("backward_q_tile", None),
        backward_kv_tile=entry.get("backward_kv_tile", None),
        schedule=entry.get("schedule", None),
        persistent=entry.get("persistent", False),
        compile=entry.get("compile", False),
        # Profiling
        bwd=entry.get("bwd", False),
        warmup_steps=entry.get("warmup_steps", 10),
        # Dry-run / optimize (not supported in batch, but needed for _run_* signature)
        dry_run=False,
        optimize=False,
        max_configs=10,
        optimize_warmup_steps=5,
        # Runtime (from CLI)
        init_mode=runtime_args.init_mode,
        memory_limit=runtime_args.memory_limit,
        seed=runtime_args.seed,
        # Output (not used by _run_*, but present for consistency)
        output_json=None,
        symbols=runtime_args.symbols,
    )


def run_batch(args):
    with open(args.input) as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise ValueError("Batch JSON input must be a list of entry objects.")

    all_results: List[ProfileResult] = []

    for i, entry in enumerate(entries):
        op = entry.get("op")
        if op is None:
            raise ValueError(f"Entry {i} missing required 'op' field.")

        print(f"[{i + 1}/{len(entries)}] Running {op} ...")

        fake_args = _entry_to_args(entry, args)

        if op == "na":
            result = _run_na(fake_args)
        elif op == "attn":
            result = _run_attn(fake_args)
        elif op == "sdpa":
            result = _run_sdpa(fake_args)
        else:
            raise ValueError(f"Entry {i}: unknown op '{op}'.")

        all_results.append(result)

        if args.print_tables:
            print_profile_table(
                result.kernels,
                use_case_str=result.use_case_str,
                symbols=args.symbols,
            )

    metadata = get_metadata()
    data = build_output_json(all_results, metadata, symbols=args.symbols)
    write_json(data, args.output)
    print(f"Results written to {args.output}")
