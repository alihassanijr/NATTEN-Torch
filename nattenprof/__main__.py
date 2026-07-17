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

"""CLI entry: parse args -> Problem.from_args -> dry-run/optimize/profile -> output."""

import torch

from nattenprof.cli import get_args
from nattenprof.engine import ProfileOptions
from nattenprof.output import (
    build_output_json,
    get_metadata,
    print_profile_table,
    write_json,
)
from nattenprof.problem import Problem


def _setup_runtime(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)


def _settings_from_args(args) -> ProfileOptions:
    return ProfileOptions(
        warmup_steps=args.warmup_steps,
        init_mode=args.init_mode,
        memory_limit=args.memory_limit,
        seed=args.seed,
    )


def _run(args):
    """Unified pipeline for na / attn / sdpa."""
    problem = Problem.from_args(args)
    settings = _settings_from_args(args)

    if getattr(args, "dry_run", False):
        problem.dry_run(max_configs=args.max_configs)
        return None

    if getattr(args, "optimize", False):
        optimize_settings = ProfileOptions(
            warmup_steps=args.optimize_warmup_steps,
            init_mode=args.init_mode,
            memory_limit=args.memory_limit,
            seed=args.seed,
        )
        problem.optimize(settings=optimize_settings)

    problem.check_config()
    return problem.profile(settings)


def main(args=None):
    if args is None:
        args = get_args()
    _setup_runtime(args)
    result = _run(args)
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


if __name__ == "__main__":
    main()
