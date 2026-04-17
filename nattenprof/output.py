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

"""Output: data classes, JSON serialization, table printing, progress bar."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

try:
    from rich.console import Console
    from rich.table import Table as RichTable

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from tqdm import tqdm  # type: ignore[import-untyped]

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

if TYPE_CHECKING:
    from nattenprof.trace import KernelType


# ---- Time formatting ----


def format_time_us(time_us: float) -> str:
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


# ---- Data classes ----


@dataclass
class KernelResult:
    kernel_type: KernelType
    namespace: str
    arch: str
    op_name: str
    symbol: str
    num_calls: int
    time_us: float

    @property
    def category(self):
        from nattenprof.trace import KERNEL_TYPE_TO_CATEGORY

        return KERNEL_TYPE_TO_CATEGORY[self.kernel_type]

    @property
    def time_str(self) -> str:
        return format_time_us(self.time_us)

    def to_dict(self, symbols: bool = False) -> dict:
        d: Dict[str, Any] = {
            "category": self.category.value,
            "kernel_type": self.kernel_type.value,
            "namespace": self.namespace,
            "arch": self.arch,
            "op_name": self.op_name,
            "num_calls": self.num_calls,
            "time_us": self.time_us,
        }
        if symbols:
            d["symbol"] = self.symbol
        return d


@dataclass
class ProfileResult:
    operation: str
    config: Dict[str, Any]
    kernels: List[KernelResult]
    use_case_str: str = ""
    optimize_info: Optional[Dict[str, Any]] = None

    def _sum_time_for_category(self, cat) -> float:
        return sum(k.time_us for k in self.kernels if k.category == cat)

    @property
    def total_time_us(self) -> float:
        return sum(k.time_us for k in self.kernels)

    @property
    def attn_fwd_time_us(self) -> float:
        from nattenprof.trace import Category

        return self._sum_time_for_category(Category.AttnFwd)

    @property
    def attn_bwd_time_us(self) -> float:
        from nattenprof.trace import Category

        return self._sum_time_for_category(Category.AttnBwd)

    @property
    def memory_ops_time_us(self) -> float:
        from nattenprof.trace import Category

        return self._sum_time_for_category(Category.MemoryOps)

    @property
    def misc_time_us(self) -> float:
        from nattenprof.trace import Category

        return self._sum_time_for_category(Category.Misc)

    def summary_dict(self, symbols: bool = False) -> dict:
        d: Dict[str, Any] = {
            "operation": self.operation,
            "config": self.config,
            "total_us": self.total_time_us,
            "breakdown": {
                "attn_fwd_us": self.attn_fwd_time_us,
                "attn_bwd_us": self.attn_bwd_time_us,
                "memory_ops_us": self.memory_ops_time_us,
                "misc_us": self.misc_time_us,
            },
            "kernels": [k.to_dict(symbols=symbols) for k in self.kernels],
        }
        if self.optimize_info is not None:
            d["optimize"] = self.optimize_info
        return d


# ---- JSON ----


def build_output_json(
    results: List[ProfileResult],
    metadata: Dict[str, str],
    symbols: bool = False,
) -> dict:
    return {
        "metadata": metadata,
        "results": [r.summary_dict(symbols=symbols) for r in results],
    }


def write_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_metadata() -> Dict[str, str]:
    meta = {
        "timestamp": datetime.now().isoformat(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name()
        meta["cuda_version"] = str(torch.version.cuda)
    try:
        import natten

        meta["natten_version"] = natten.__version__
    except Exception:
        pass
    return meta


# ---- Table printing ----


def get_terminal_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 120


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _print_rich_table(
    title: Optional[str],
    headers: List[str],
    rows: List[List[str]],
    footer: Optional[List[str]] = None,
):
    table = RichTable(show_header=True, header_style="bold", title=title)
    for h in headers:
        table.add_column(h, justify="center")
    for r in rows:
        table.add_row(*r)
    if footer is not None:
        table.add_section()
        table.add_row(*footer, style="bold")
    Console().print(table)


def _bold(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return text


def _print_ascii_table(
    title: Optional[str],
    headers: List[str],
    rows: List[List[str]],
    footer: Optional[List[str]] = None,
):
    all_rows = [headers] + rows + ([footer] if footer else [])
    columns = list(zip(*all_rows))
    col_widths = [max(len(str(cell)) for cell in col) for col in columns]

    def format_row(row, bold_row=False):
        formatted = " | ".join(
            str(cell).ljust(width) for cell, width in zip(row, col_widths)
        )
        formatted = f"| {formatted} |"
        return _bold(formatted) if bold_row else formatted

    def separator(char="-", junction="+"):
        return junction + junction.join(char * (w + 2) for w in col_widths) + junction

    if title:
        title_lines = title.split("\n")
        total_width = sum(col_widths) + 3 * len(col_widths) + 1
        print("|" + "=" * (total_width - 2) + "|")
        for t in title_lines:
            print("|" + str(t).center(total_width - 2) + "|")
        print("|" + "=" * (total_width - 2) + "|")

    print(separator("="))
    print(format_row(headers, bold_row=True))
    print(separator("="))

    for row in rows:
        print(format_row(row))

    if footer:
        print(separator("="))
        print(format_row(footer, bold_row=True))

    print(separator("="))
    print()


def print_table(
    title: Optional[str],
    headers: List[str],
    values: List[List[str]],
    has_footer: bool = False,
):
    if has_footer and len(values) > 0:
        rows = values[:-1]
        footer = values[-1]
    else:
        rows = values
        footer = None

    if HAS_RICH:
        _print_rich_table(title, headers, rows, footer)
    else:
        _print_ascii_table(title, headers, rows, footer)


def print_profile_table(
    results: List[KernelResult],
    use_case_str: Optional[str] = None,
    symbols: bool = False,
):
    """Print the breakdown table + category summary."""
    if use_case_str is not None:
        print(use_case_str)
        print()

    term_width = get_terminal_width()
    symbol_max = max(30, term_width - 100)

    headers = [
        "Namespace",
        "Category",
        "Arch",
        "Operation",
        "# calls",
        "Runtime",
    ]
    if symbols:
        headers.append("Symbol")

    values = []
    for r in results:
        row = [
            r.namespace,
            r.category.name,
            r.arch,
            r.op_name,
            str(r.num_calls),
            r.time_str,
        ]
        if symbols:
            row.append(truncate(r.symbol, symbol_max))
        values.append(row)

    ncols = len(headers)
    rt_idx = headers.index("Runtime")
    op_idx = headers.index("Operation")

    total_us = sum(r.time_us for r in results)
    footer = [""] * ncols
    footer[op_idx] = "Total"
    footer[rt_idx] = format_time_us(total_us)
    values.append(footer)

    print_table("Breakdown", headers, values, has_footer=True)

    # Per-category summary
    from nattenprof.trace import CATEGORY_PRIORITY

    cat_totals: Dict[Any, float] = {}
    for r in results:
        c = r.category
        if c not in cat_totals:
            cat_totals[c] = 0.0
        cat_totals[c] += r.time_us

    if len(cat_totals) > 1:
        summary_headers = ["Category", "Runtime"]
        summary_values = []
        for c in sorted(cat_totals, key=lambda x: CATEGORY_PRIORITY.get(x, 99)):
            summary_values.append([c.name, format_time_us(cat_totals[c])])
        summary_values.append(["Total", format_time_us(total_us)])
        print_table("Summary", summary_headers, summary_values, has_footer=True)


# ---- Progress bar ----


def progress_bar(iterable, total: int):
    if HAS_TQDM:
        return tqdm(iterable, total=total)
    return iterable
