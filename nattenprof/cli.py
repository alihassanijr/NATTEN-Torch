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

import argparse

DTYPE_CHOICES = ["fp32", "bf16", "fp16", "e4m3", "e5m2"]

NATTEN_FNA_BACKENDS = ["cutlass-fna", "blackwell-fna", "hopper-fna", "flex-fna"]
NATTEN_FMHA_BACKENDS = ["cutlass-fmha", "blackwell-fmha", "hopper-fmha", "flex-fmha"]
SDPA_BACKENDS = ["xformers", "cudnn", "fav2"]

SCHEDULE_CHOICES = ["non", "coop", "pp"]
INIT_MODE_CHOICES = ["randn", "uniform", "ones"]


def _make_shared_parent() -> argparse.ArgumentParser:
    """Shared parent parser inherited by all subcommands via parents=[]."""
    parent = argparse.ArgumentParser(add_help=False)

    # -- Problem-shape flags --
    parent.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="QKV batch size.",
    )
    parent.add_argument(
        "-n",
        "--heads",
        type=int,
        default=1,
        help="Number of Q heads.",
    )
    parent.add_argument(
        "--heads-kv",
        type=int,
        default=None,
        help="Number of KV heads (GQA/MQA). Defaults to --heads.",
    )
    parent.add_argument(
        "-d",
        "--dim",
        type=int,
        default=64,
        help="QK head dim.",
    )
    parent.add_argument(
        "--dim-value",
        type=int,
        default=None,
        help="V head dim, if different from QK. Defaults to --dim.",
    )
    parent.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=DTYPE_CHOICES,
        help="Element (data) type.",
    )
    parent.add_argument(
        "--bwd",
        action="store_true",
        help="Profile backward pass as well as forward pass.",
    )
    parent.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup iterations.",
    )

    # -- Runtime flags --
    parent.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results as JSON to this file path.",
    )
    parent.add_argument(
        "--symbols",
        action="store_true",
        help="Include raw kernel symbol names in output (table + JSON).",
    )
    parent.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA GPU device index.",
    )
    parent.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable torch.use_deterministic_algorithms(True).",
    )
    parent.add_argument(
        "--init-mode",
        type=str,
        default="randn",
        choices=INIT_MODE_CHOICES,
        help="Tensor initialization mode.",
    )
    parent.add_argument(
        "--memory-limit",
        type=float,
        default=10.0,
        help="Max GPU memory (GB) for tensor pool.",
    )
    parent.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for tensor generation.",
    )

    return parent


def _add_tile_args_shape(parser: argparse.ArgumentParser):
    """Tile args as N-D shapes (for NA subcommand)."""
    parser.add_argument(
        "--q-tile",
        type=int,
        nargs="*",
        default=None,
        help="Q tile shape in the forward pass kernel.",
    )
    parser.add_argument(
        "--kv-tile",
        type=int,
        nargs="*",
        default=None,
        help="KV tile shape in the forward pass kernel.",
    )
    parser.add_argument(
        "--backward-q-tile",
        type=int,
        nargs="*",
        default=None,
        help="Q tile shape in the backward pass kernel.",
    )
    parser.add_argument(
        "--backward-kv-tile",
        type=int,
        nargs="*",
        default=None,
        help="KV tile shape in the backward pass kernel.",
    )


def _add_tile_args_scalar(parser: argparse.ArgumentParser):
    """Tile args as scalar sizes (for attn subcommand)."""
    parser.add_argument(
        "--q-tile",
        type=int,
        default=None,
        help="Q tile size in the forward pass kernel.",
    )
    parser.add_argument(
        "--kv-tile",
        type=int,
        default=None,
        help="KV tile size in the forward pass kernel.",
    )
    parser.add_argument(
        "--backward-q-tile",
        type=int,
        default=None,
        help="Q tile size in the backward pass kernel.",
    )
    parser.add_argument(
        "--backward-kv-tile",
        type=int,
        default=None,
        help="KV tile size in the backward pass kernel.",
    )


def _add_backend_perf_args(parser: argparse.ArgumentParser):
    """Schedule, persistent, compile flags."""
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        choices=SCHEDULE_CHOICES,
        help="Kernel schedule (hopper only): non, coop, pp.",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use persistent scheduling (blackwell only).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile flex attention mask + kernel.",
    )


def _add_optimize_args(parser: argparse.ArgumentParser):
    """Dry-run and optimize flags."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display valid configurations and exit.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Search for best configuration.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=10,
        help="Max configs to display in dry-run. 0 = show all.",
    )
    parser.add_argument(
        "--optimize-warmup-steps",
        type=int,
        default=5,
        help="Warmup steps for optimize search.",
    )


def _build_na_parser(subparsers, parent):
    na = subparsers.add_parser(
        "na",
        help="Profile neighborhood attention.",
        parents=[parent],
    )

    na.add_argument(
        "-i",
        "--input-size",
        type=int,
        nargs="+",
        required=True,
        help="Token layout shape (1-3 ints).",
    )
    na.add_argument(
        "-w",
        "--window-size",
        type=int,
        nargs="*",
        default=None,
        help="Window size (kernel_size). Defaults to --input-size (self attention).",
    )
    na.add_argument(
        "-s",
        "--stride",
        type=int,
        nargs="*",
        default=None,
        help="Stride. Defaults to 1s.",
    )
    na.add_argument(
        "--dilation",
        type=int,
        nargs="*",
        default=None,
        help="Dilation. Defaults to 1s.",
    )
    na.add_argument(
        "-c",
        "--causal",
        type=bool,
        nargs="*",
        default=None,
        help="Causal mask per dimension. Defaults to all False.",
    )
    na.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=NATTEN_FNA_BACKENDS,
        help="FNA backend.",
    )
    na.add_argument(
        "--fmha-backend",
        type=str,
        default=None,
        choices=NATTEN_FMHA_BACKENDS,
        help="FMHA backend for self-attn fast path / cross-attn.",
    )
    na.add_argument(
        "--add-kv",
        type=int,
        default=0,
        help="Number of additional KV tokens.",
    )

    _add_tile_args_shape(na)
    _add_backend_perf_args(na)
    _add_optimize_args(na)

    return na


def _add_seqlen_args(parser: argparse.ArgumentParser):
    """Sequence length args shared by attn and sdpa."""
    parser.add_argument(
        "-i",
        "--input-size",
        type=int,
        nargs="+",
        default=None,
        help="Token layout shape. Seqlen = product of dims. Sets both Q and KV.",
    )
    parser.add_argument(
        "-q",
        "--seqlen",
        type=int,
        default=None,
        help="Q sequence length. Overrides -i for Q side.",
    )
    parser.add_argument(
        "-k",
        "--seqlen-kv",
        type=int,
        default=None,
        help="KV sequence length. Overrides -i for KV side.",
    )


def _build_attn_parser(subparsers, parent):
    attn = subparsers.add_parser(
        "attn",
        help="Profile NATTEN standard attention.",
        parents=[parent],
    )

    _add_seqlen_args(attn)
    attn.add_argument(
        "--is-causal",
        action="store_true",
        help="Enable causal mask.",
    )
    attn.add_argument(
        "--varlen",
        action="store_true",
        help="Variable-length mode.",
    )
    attn.add_argument(
        "--seqlens",
        type=int,
        nargs="+",
        default=None,
        help="Per-batch Q sequence lengths (requires --varlen).",
    )
    attn.add_argument(
        "--seqlens-kv",
        type=int,
        nargs="+",
        default=None,
        help="Per-batch KV sequence lengths (requires --varlen).",
    )
    attn.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=NATTEN_FMHA_BACKENDS,
        help="FMHA backend.",
    )

    _add_tile_args_scalar(attn)
    _add_backend_perf_args(attn)
    _add_optimize_args(attn)

    return attn


def _build_sdpa_parser(subparsers, parent):
    sdpa = subparsers.add_parser(
        "sdpa",
        help="Profile torch SDPA baseline.",
        parents=[parent],
    )

    _add_seqlen_args(sdpa)
    sdpa.add_argument(
        "--is-causal",
        action="store_true",
        help="Enable causal mask.",
    )
    sdpa.add_argument(
        "--backend",
        type=str,
        default="cudnn",
        choices=SDPA_BACKENDS,
        help="SDPA backend.",
    )

    return sdpa


def _build_batch_parser(subparsers, parent):
    batch = subparsers.add_parser(
        "batch",
        help="Batch mode: run from JSON config file.",
        parents=[parent],
    )

    batch.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON input config file.",
    )
    batch.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to JSON output results file.",
    )
    batch.add_argument(
        "--print",
        action="store_true",
        dest="print_tables",
        help="Also print tables to terminal while running.",
    )

    return batch


def get_args():
    parent = _make_shared_parent()

    parser = argparse.ArgumentParser(
        description="nattenprof: NATTEN profiling toolkit.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    _build_na_parser(subparsers, parent)
    _build_attn_parser(subparsers, parent)
    _build_sdpa_parser(subparsers, parent)
    _build_batch_parser(subparsers, parent)

    args = parser.parse_args()

    # Defaults
    if not hasattr(args, "heads_kv") or args.heads_kv is None:
        args.heads_kv = args.heads
    if not hasattr(args, "dim_value") or args.dim_value is None:
        args.dim_value = args.dim

    # Resolve seqlen from -i/--input-size for attn/sdpa
    if args.subcommand in ("attn", "sdpa"):
        import math

        if args.input_size is not None:
            if args.seqlen is not None or args.seqlen_kv is not None:
                parser.error(
                    "-i/--input-size cannot be used together with"
                    " -q/--seqlen or -k/--seqlen-kv."
                )
            seqlen_from_input = math.prod(args.input_size)
            args.seqlen = seqlen_from_input
            args.seqlen_kv = seqlen_from_input

        if args.seqlen is None:
            parser.error(
                f"{args.subcommand}: either -i/--input-size or -q/--seqlen is required."
            )

        if args.seqlen_kv is None:
            args.seqlen_kv = args.seqlen

    return args
