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

"""Unit tests for nattenprof via JSON batch interface."""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

import pytest
import torch

# TODO(relocate): Remove VENV_PYTHON/REPO_ROOT plumbing once nattenprof lives under
# src/natten/profiler/ and its tests under tests/. The subprocess + PYTHONPATH
# dance is only needed because nattenprof currently sits at the repo root.
VENV_PYTHON = sys.executable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

HAS_CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


def _run_batch(entries: List[Dict[str, Any]], extra_args: List[str] = None):
    """Run nattenprof batch and return parsed JSON output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f_in:
        json.dump(entries, f_in)
        input_path = f_in.name

    output_path = input_path.replace(".json", "_out.json")

    try:
        cmd = [
            VENV_PYTHON,
            "-m",
            "nattenprof",
            "batch",
            "--input",
            input_path,
            "--output",
            output_path,
            "--symbols",
        ]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        # TODO(relocate): Drop this PYTHONPATH injection once nattenprof lives
        # under src/natten/profiler/ — it is only here because we run from repo root.
        env["PYTHONPATH"] = f"src:.:{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            # TODO(relocate): cwd=REPO_ROOT is only needed while nattenprof lives
            # at repo root. After relocation, remove this.
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nattenprof failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        with open(output_path) as f:
            return json.load(f)

    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.unlink(p)


def _check_result_structure(result: dict):
    """Assert a single result entry has correct structure."""
    assert "operation" in result
    assert "config" in result
    assert "total_us" in result
    assert "breakdown" in result
    bd = result["breakdown"]
    for key in [
        "attn_fwd_us",
        "attn_bwd_us",
        "memory_ops_us",
        "misc_us",
    ]:
        assert key in bd
    assert "kernels" in result
    assert len(result["kernels"]) > 0
    for k in result["kernels"]:
        assert "category" in k
        assert "kernel_type" in k
        assert "namespace" in k
        assert "arch" in k
        assert "op_name" in k
        assert "symbol" in k
        assert "num_calls" in k
        assert "time_us" in k


# ---- Metadata ----


@skip_no_cuda
def test_metadata():
    data = _run_batch(
        [{"op": "sdpa", "seqlen": 512, "dim": 64, "dtype": "bf16", "backend": "cudnn"}]
    )
    meta = data["metadata"]
    assert "timestamp" in meta
    assert "torch_version" in meta
    assert "gpu" in meta
    assert "cuda_version" in meta


# ---- SDPA tests ----


@skip_no_cuda
def test_sdpa_cudnn():
    data = _run_batch(
        [{"op": "sdpa", "seqlen": 512, "dim": 64, "dtype": "bf16", "backend": "cudnn"}]
    )
    assert len(data["results"]) == 1
    r = data["results"][0]
    _check_result_structure(r)
    assert r["operation"] == "sdpa"
    assert r["total_us"] > 0


@skip_no_cuda
def test_sdpa_fav2():
    data = _run_batch(
        [{"op": "sdpa", "seqlen": 512, "dim": 64, "dtype": "fp16", "backend": "fav2"}]
    )
    r = data["results"][0]
    _check_result_structure(r)
    assert r["total_us"] > 0


@skip_no_cuda
def test_sdpa_different_kv_seqlen():
    data = _run_batch(
        [
            {
                "op": "sdpa",
                "seqlen": 512,
                "seqlen_kv": 256,
                "dim": 64,
                "dtype": "bf16",
                "backend": "cudnn",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)


@skip_no_cuda
@pytest.mark.xfail(
    reason="cuDNN/Flash/xformers SDPA require heads == heads_kv for dense input",
    strict=True,
)
def test_sdpa_gqa():
    _run_batch(
        [
            {
                "op": "sdpa",
                "seqlen": 512,
                "dim": 64,
                "heads": 8,
                "heads_kv": 2,
                "dtype": "bf16",
                "backend": "cudnn",
            }
        ]
    )


# ---- NA tests ----


@skip_no_cuda
def test_na_1d_hopper():
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)
    assert r["operation"] == "na"
    # Should have attention_forward + tok_perm
    ktypes = {k["kernel_type"] for k in r["kernels"]}
    assert "attention_forward" in ktypes
    assert "tok_perm" in ktypes


@skip_no_cuda
def test_na_2d_hopper():
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [64, 64],
                "window_size": [32, 32],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)
    ktypes = {k["kernel_type"] for k in r["kernels"]}
    assert "attention_forward" in ktypes


@skip_no_cuda
def test_na_3d_hopper():
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [16, 16, 16],
                "window_size": [8, 8, 8],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)


@skip_no_cuda
def test_na_self_attention_fallthrough():
    """When window_size == input_size, NA falls through to FMHA."""
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "dim": 128,
                "dtype": "bf16",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)
    # Should be FMHA, not FNA (self-attn fast path)
    for k in r["kernels"]:
        if k["kernel_type"] == "attention_forward":
            assert k["op_name"] == "FMHAForward"


@skip_no_cuda
def test_na_strided():
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "stride": [128],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)


@skip_no_cuda
def test_na_bwd():
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
                "bwd": True,
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)
    assert r["breakdown"]["attn_bwd_us"] > 0
    ktypes = {k["kernel_type"] for k in r["kernels"]}
    assert "attention_backward" in ktypes


@skip_no_cuda
def test_na_cutlass():
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "dim": 128,
                "dtype": "bf16",
                "backend": "cutlass-fna",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)


# ---- Attention tests ----


@skip_no_cuda
def test_attn_hopper():
    data = _run_batch(
        [
            {
                "op": "attn",
                "seqlen": 1024,
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fmha",
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)
    assert r["operation"] == "attn"


@skip_no_cuda
def test_attn_cutlass_causal():
    data = _run_batch(
        [
            {
                "op": "attn",
                "seqlen": 1024,
                "dim": 128,
                "dtype": "bf16",
                "backend": "cutlass-fmha",
                "is_causal": True,
            }
        ]
    )
    r = data["results"][0]
    _check_result_structure(r)


# ---- Batch multi-entry ----


@skip_no_cuda
def test_batch_multi():
    data = _run_batch(
        [
            {
                "op": "sdpa",
                "seqlen": 512,
                "dim": 64,
                "dtype": "bf16",
                "backend": "cudnn",
            },
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
            },
            {
                "op": "attn",
                "seqlen": 1024,
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fmha",
            },
        ]
    )
    assert len(data["results"]) == 3
    assert data["results"][0]["operation"] == "sdpa"
    assert data["results"][1]["operation"] == "na"
    assert data["results"][2]["operation"] == "attn"


# ---- Reproducibility ----


@skip_no_cuda
def test_reproducibility_same_kernels():
    """Run same config twice, verify same kernels reported and runtimes in range."""
    entry = {
        "op": "na",
        "input_size": [1024],
        "window_size": [256],
        "dim": 128,
        "dtype": "bf16",
        "backend": "hopper-fna",
    }
    data1 = _run_batch([entry])
    data2 = _run_batch([entry])

    r1 = data1["results"][0]
    r2 = data2["results"][0]

    # Same kernels
    k1_names = [(k["op_name"], k["num_calls"]) for k in r1["kernels"]]
    k2_names = [(k["op_name"], k["num_calls"]) for k in r2["kernels"]]
    assert k1_names == k2_names

    # Runtimes within 50% of each other (profiler variance is real)
    t1 = r1["total_us"]
    t2 = r2["total_us"]
    ratio = max(t1, t2) / max(min(t1, t2), 0.001)
    assert ratio < 1.5, f"Runtimes too far apart: {t1} vs {t2}"


# ---- Breakdown consistency ----


@skip_no_cuda
def test_breakdown_sums():
    """Verify breakdown components sum to total."""
    data = _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "dim": 128,
                "dtype": "bf16",
                "backend": "hopper-fna",
                "bwd": True,
            }
        ]
    )
    r = data["results"][0]
    bd = r["breakdown"]
    component_sum = sum(bd.values())
    # Should be close to total (floating point)
    assert abs(component_sum - r["total_us"]) < 0.01


# ---- Expected failures ----


@skip_no_cuda
@pytest.mark.xfail(reason="Invalid op type should raise", strict=True)
def test_invalid_op():
    _run_batch([{"op": "bogus", "seqlen": 512, "dim": 64}])


@skip_no_cuda
@pytest.mark.xfail(reason="Missing required field should raise", strict=True)
def test_missing_input_size():
    _run_batch([{"op": "na", "dim": 64, "dtype": "bf16", "backend": "hopper-fna"}])


@skip_no_cuda
@pytest.mark.xfail(reason="Invalid SDPA backend should raise", strict=True)
def test_invalid_sdpa_backend():
    _run_batch(
        [{"op": "sdpa", "seqlen": 512, "dim": 64, "dtype": "bf16", "backend": "bogus"}]
    )


@skip_no_cuda
@pytest.mark.xfail(reason="Invalid FNA backend should raise", strict=True)
def test_invalid_fna_backend():
    _run_batch(
        [
            {
                "op": "na",
                "input_size": [1024],
                "window_size": [256],
                "dim": 128,
                "dtype": "bf16",
                "backend": "bogus-fna",
            }
        ]
    )


@skip_no_cuda
@pytest.mark.xfail(reason="Window size larger than input should raise", strict=True)
def test_window_larger_than_input():
    _run_batch(
        [
            {
                "op": "na",
                "input_size": [128],
                "window_size": [256],
                "dim": 64,
                "dtype": "bf16",
                "backend": "hopper-fna",
            }
        ]
    )
