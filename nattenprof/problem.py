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

"""Problem hierarchy: AttnProblem + NAProblem + SdpaProblem, plus PerfConfig.

Compute shape lives on Problem; kernel dispatch + tuning knobs live on a
separate PerfConfig attached as `problem.perf`. Subclasses override `_run_op`
(which kernel to call); the profile() loop itself is shared.

PerfConfig field names mirror natten's argument names exactly:
  - *_tile_shape + fna_backend: NA-native group (FNA kernels).
  - *_tile_size  + fmha_backend: FMHA group (attention / SDPA / NA self-attn routing).

Mode invariants (enforced in each Problem's check_config):
  - AttnProblem: fna_backend None; all *_tile_shape None; fmha_backend set; sizes may be set.
  - SdpaProblem: fna_backend None; all tile fields None; fmha_backend set.
  - NAProblem:   BOTH fna_backend and fmha_backend set; BOTH *_tile_shape and *_tile_size set.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from natten.types import KernelSchedule
from torch import Tensor

from nattenprof.output import format_section

DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
DTYPE_SHORT = {v: k for k, v in DTYPE_MAP.items()}

KERNEL_SCHEDULE_NAMES = {
    KernelSchedule.NonPersistent: "Non-persistent",
    KernelSchedule.WarpSpecializedCooperative: "WS Cooperative",
    KernelSchedule.WarpSpecializedPingpong: "WS Ping-ponging",
}


# Backend-specific perf knobs
# kernel_schedule applies only to hopper-* backends
_SCHEDULE_BACKENDS = {"hopper-fmha", "hopper-fna"}
# is_persistent applies only to blackwell-* backends
_PERSISTENT_BACKENDS = {"blackwell-fmha", "blackwell-fna"}
# torch_compile applies only to flex-* backends
_COMPILE_BACKENDS = {"flex-fmha", "flex-fna"}


# =============================================================================
# PerfConfig — kernel dispatch + tuning knobs (mode-agnostic storage)
# =============================================================================


@dataclass(kw_only=True)
class PerfConfig:
    """Kernel dispatch + tuning knobs. Attached to Problem as `problem.perf`.

    Fields mirror natten's parameter names. Validation of which fields apply
    in which mode lives in each Problem subclass's check_config, not here.
    """

    # FNA group (NA only)
    fna_backend: Optional[str] = None
    q_tile_shape: Optional[Tuple[int, ...]] = None
    kv_tile_shape: Optional[Tuple[int, ...]] = None
    backward_q_tile_shape: Optional[Tuple[int, ...]] = None
    backward_kv_tile_shape: Optional[Tuple[int, ...]] = None

    # FMHA group (Attn, Sdpa, NA self-attn routing)
    fmha_backend: Optional[str] = None
    q_tile_size: Optional[int] = None
    kv_tile_size: Optional[int] = None
    backward_q_tile_size: Optional[int] = None
    backward_kv_tile_size: Optional[int] = None

    # Shared knobs
    kernel_schedule: Optional[str] = None
    is_persistent: bool = False
    torch_compile: bool = False

    def __str__(self) -> str:
        """Multi-line "Performance knobs:" section. Content-driven line wrap."""
        segments: List[str] = []

        # FNA group
        if self.fna_backend is not None:
            segments.append(f"fna_backend={self.fna_backend}")
        if self.q_tile_shape is not None:
            segments.append(f"q_tile_shape={self.q_tile_shape}")
        if self.kv_tile_shape is not None:
            segments.append(f"kv_tile_shape={self.kv_tile_shape}")
        if self.backward_q_tile_shape is not None:
            segments.append(f"backward_q_tile_shape={self.backward_q_tile_shape}")
        if self.backward_kv_tile_shape is not None:
            segments.append(f"backward_kv_tile_shape={self.backward_kv_tile_shape}")

        # FMHA group
        if self.fmha_backend is not None:
            segments.append(f"fmha_backend={self.fmha_backend}")
        if self.q_tile_size is not None:
            segments.append(f"q_tile_size={self.q_tile_size}")
        if self.kv_tile_size is not None:
            segments.append(f"kv_tile_size={self.kv_tile_size}")
        if self.backward_q_tile_size is not None:
            segments.append(f"backward_q_tile_size={self.backward_q_tile_size}")
        if self.backward_kv_tile_size is not None:
            segments.append(f"backward_kv_tile_size={self.backward_kv_tile_size}")

        # Shared (conditional on which backend(s) support the knob)
        active = {self.fna_backend, self.fmha_backend}
        if active & _SCHEDULE_BACKENDS:
            sched = KERNEL_SCHEDULE_NAMES.get(
                self.kernel_schedule, str(self.kernel_schedule)
            )
            segments.append(f"kernel_schedule={sched}")
        if active & _PERSISTENT_BACKENDS:
            segments.append(f"is_persistent={self.is_persistent}")
        if active & _COMPILE_BACKENDS:
            segments.append(f"torch_compile={self.torch_compile}")

        return format_section("Performance knobs:", segments)

    # ---- Factory ----
    @classmethod
    def from_args(cls, args) -> "PerfConfig":
        """Build a PerfConfig from a CLI argparse namespace, per-subcommand."""
        shared = {
            "kernel_schedule": getattr(args, "schedule", None),
            "is_persistent": getattr(args, "persistent", False),
            "torch_compile": getattr(args, "compile", False),
        }
        if args.subcommand == "sdpa":
            return cls(fmha_backend=getattr(args, "backend", None), **shared)
        if args.subcommand == "attn":
            return cls(
                fmha_backend=getattr(args, "backend", None),
                q_tile_size=getattr(args, "q_tile", None),
                kv_tile_size=getattr(args, "kv_tile", None),
                backward_q_tile_size=getattr(args, "backward_q_tile", None),
                backward_kv_tile_size=getattr(args, "backward_kv_tile", None),
                **shared,
            )
        if args.subcommand == "na":
            q_shape = tuple(args.q_tile) if args.q_tile else None
            kv_shape = tuple(args.kv_tile) if args.kv_tile else None
            bwd_q_shape = tuple(args.backward_q_tile) if args.backward_q_tile else None
            bwd_kv_shape = (
                tuple(args.backward_kv_tile) if args.backward_kv_tile else None
            )
            perf = cls(
                fna_backend=getattr(args, "backend", None),
                fmha_backend=getattr(args, "fmha_backend", None),
                q_tile_shape=q_shape,
                kv_tile_shape=kv_shape,
                backward_q_tile_shape=bwd_q_shape,
                backward_kv_tile_shape=bwd_kv_shape,
                **shared,
            )
            perf.derive_sizes_from_shapes()
            return perf
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    # ---- Validation (per-Problem invariants) ----
    def validate_attn_mode(self) -> None:
        """AttnProblem: FNA group must be unset."""
        assert self.fna_backend is None, "AttnProblem must not set fna_backend."
        assert all(
            x is None
            for x in (
                self.q_tile_shape,
                self.kv_tile_shape,
                self.backward_q_tile_shape,
                self.backward_kv_tile_shape,
            )
        ), "AttnProblem must not set FNA tile shapes."

    def validate_sdpa_mode(self) -> None:
        """SdpaProblem: FNA group + all tile fields must be unset."""
        assert self.fna_backend is None, "SdpaProblem must not set fna_backend."
        assert all(
            x is None
            for x in (
                self.q_tile_shape,
                self.kv_tile_shape,
                self.backward_q_tile_shape,
                self.backward_kv_tile_shape,
                self.q_tile_size,
                self.kv_tile_size,
                self.backward_q_tile_size,
                self.backward_kv_tile_size,
            )
        ), "SdpaProblem must not set any tile fields."

    # ---- Resolve / normalize (in-place) ----
    def normalize_schedule(self) -> None:
        from natten.utils.checks import check_kernel_schedule

        self.kernel_schedule = check_kernel_schedule(self.kernel_schedule)

    def resolve_sdpa_default(self) -> None:
        """Set the SDPA default backend if none specified."""
        if self.fmha_backend is None:
            self.fmha_backend = "cudnn"

    def resolve_fmha(
        self, q, *, is_causal, is_varlen, validate_tiles: bool = True
    ) -> None:
        """Resolve fmha_backend from q; optionally validate + normalize FMHA tiles.

        `validate_tiles=False` is used by NA non-self-attn, where the FMHA path
        won't actually run and tile sizes will be derived from FNA shapes instead.
        """
        from natten.backends import choose_fmha_backend

        self.fmha_backend = self.fmha_backend or choose_fmha_backend(
            q,
            q,
            q,
            is_causal=is_causal,
            is_varlen=is_varlen,
            torch_compile=self.torch_compile,
        )
        if validate_tiles:
            self._validate_fmha_tiles(q)

    def _validate_fmha_tiles(self, q) -> None:
        from natten.backends.configs.cutlass import (
            check_cutlass_fmha_backward_config,
            check_cutlass_fmha_forward_config,
        )
        from natten.backends.configs.cutlass_blackwell import (
            check_cutlass_blackwell_fmha_backward_config,
            check_cutlass_blackwell_fmha_forward_config,
        )
        from natten.backends.configs.cutlass_hopper import (
            check_cutlass_hopper_fmha_backward_config,
            check_cutlass_hopper_fmha_forward_config,
        )

        if self.fmha_backend == "cutlass-fmha":
            self.q_tile_size, self.kv_tile_size = check_cutlass_fmha_forward_config(
                q, self.q_tile_size, self.kv_tile_size
            )
            self.backward_q_tile_size, self.backward_kv_tile_size = (
                check_cutlass_fmha_backward_config(
                    q, self.backward_q_tile_size, self.backward_kv_tile_size
                )
            )
        elif self.fmha_backend == "hopper-fmha":
            (self.q_tile_size, self.kv_tile_size), self.kernel_schedule = (
                check_cutlass_hopper_fmha_forward_config(
                    q, self.q_tile_size, self.kv_tile_size, self.kernel_schedule
                )
            )
            self.backward_q_tile_size, self.backward_kv_tile_size = (
                check_cutlass_hopper_fmha_backward_config(
                    q, self.backward_q_tile_size, self.backward_kv_tile_size
                )
            )
        elif self.fmha_backend == "blackwell-fmha":
            self.q_tile_size, self.kv_tile_size = (
                check_cutlass_blackwell_fmha_forward_config(
                    q, self.q_tile_size, self.kv_tile_size
                )
            )
            self.backward_q_tile_size, self.backward_kv_tile_size = (
                check_cutlass_blackwell_fmha_backward_config(
                    q, self.backward_q_tile_size, self.backward_kv_tile_size
                )
            )
        # flex-fmha: skip (different check contract)

    def resolve_fna(self, q, *, dilation) -> None:
        """Resolve fna_backend from q + validate FNA tile shapes."""
        from natten.backends import choose_backend

        self.fna_backend = self.fna_backend or choose_backend(
            q, q, q, torch_compile=self.torch_compile
        )
        self._validate_fna_tiles(q, dilation)

    def _validate_fna_tiles(self, q, dilation) -> None:
        from natten.backends.configs.cutlass import (
            check_cutlass_fna_backward_config,
            check_cutlass_fna_forward_config,
        )
        from natten.backends.configs.cutlass_blackwell import (
            check_cutlass_blackwell_fna_backward_config,
            check_cutlass_blackwell_fna_forward_config,
        )
        from natten.backends.configs.cutlass_hopper import (
            check_cutlass_hopper_fna_backward_config,
            check_cutlass_hopper_fna_forward_config,
        )

        if self.fna_backend == "cutlass-fna":
            self.q_tile_shape, self.kv_tile_shape = check_cutlass_fna_forward_config(
                q, dilation, self.q_tile_shape, self.kv_tile_shape
            )
            self.backward_q_tile_shape, self.backward_kv_tile_shape = (
                check_cutlass_fna_backward_config(
                    q, self.backward_q_tile_shape, self.backward_kv_tile_shape
                )
            )
        elif self.fna_backend == "hopper-fna":
            (self.q_tile_shape, self.kv_tile_shape), self.kernel_schedule = (
                check_cutlass_hopper_fna_forward_config(
                    q, self.q_tile_shape, self.kv_tile_shape, self.kernel_schedule
                )
            )
            self.backward_q_tile_shape, self.backward_kv_tile_shape = (
                check_cutlass_hopper_fna_backward_config(
                    q, self.backward_q_tile_shape, self.backward_kv_tile_shape
                )
            )
        elif self.fna_backend == "blackwell-fna":
            self.q_tile_shape, self.kv_tile_shape = (
                check_cutlass_blackwell_fna_forward_config(
                    q, self.q_tile_shape, self.kv_tile_shape
                )
            )
            self.backward_q_tile_shape, self.backward_kv_tile_shape = (
                check_cutlass_blackwell_fna_backward_config(
                    q, self.backward_q_tile_shape, self.backward_kv_tile_shape
                )
            )
        # flex-fna: skip (different check contract)

    def derive_sizes_from_shapes(self) -> None:
        """Populate *_tile_size from *_tile_shape via math.prod.

        Used in NA mode to keep both groups consistent when only shapes were set.
        """
        if self.q_tile_shape is not None:
            self.q_tile_size = math.prod(self.q_tile_shape)
        if self.kv_tile_shape is not None:
            self.kv_tile_size = math.prod(self.kv_tile_shape)
        if self.backward_q_tile_shape is not None:
            self.backward_q_tile_size = math.prod(self.backward_q_tile_shape)
        if self.backward_kv_tile_shape is not None:
            self.backward_kv_tile_size = math.prod(self.backward_kv_tile_shape)


# =============================================================================
# Problem base class
# =============================================================================


@dataclass(kw_only=True)
class Problem(ABC):
    # Compute
    batch_size: int
    heads: int
    heads_kv: int
    dim: int
    dim_value: int
    dtype: torch.dtype
    bwd: bool = False

    # Kernel dispatch + tuning. See PerfConfig for field layout.
    perf: PerfConfig = field(default_factory=PerfConfig)

    # ---- Factory ----
    @classmethod
    def from_args(cls, args) -> "Problem":
        if args.subcommand == "na":
            return NAProblem._from_args(args)
        if args.subcommand == "attn":
            return AttnProblem._from_args(args)
        if args.subcommand == "sdpa":
            return SdpaProblem._from_args(args)
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    # ---- Abstract: subclass must provide ----
    @abstractmethod
    def get_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def _run_op(self, tensors: Dict[str, Tensor]) -> None: ...

    @abstractmethod
    def _config_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def _operation_name(self) -> str: ...

    # ---- Default lifecycle hooks; subclass overrides as needed ----
    def check_config(self) -> None:
        """Resolve backend / tiles in-place. Default: no-op."""
        pass

    def dry_run(self, max_configs: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support dry-run.")

    def optimize(self, settings) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support optimize.")

    # ---- Shared profile loop ----
    def profile(self, settings) -> "Any":
        """Warmup + profiler capture. Subclasses only override _run_op."""
        from nattenprof.engine import profile_op
        from nattenprof.output import ProfileResult
        from nattenprof.tensors import InitMode, TensorPool

        torch.set_grad_enabled(self.bwd)

        pool = TensorPool(
            shapes=self.get_tensor_shapes(),
            dtype=self.dtype,
            device=torch.device("cuda"),
            init_mode=InitMode(settings.init_mode),
            memory_limit_gb=settings.memory_limit,
            seed=settings.seed,
            requires_grad=self.bwd,
        )

        results = profile_op(
            pool=pool,
            run_fn=self._run_op,
            warmup_steps=settings.warmup_steps,
        )
        if not results:
            raise RuntimeError("Profiler captured no kernel events after retries.")

        profiler_section = format_section(
            "Profiler knobs:",
            [f"warmup_steps={settings.warmup_steps}", "profiling_steps=1"],
        )
        use_case = "\n\n".join([str(self), str(self.perf), profiler_section])
        return ProfileResult(
            operation=self._operation_name(),
            config=self._config_dict(),
            kernels=results,
            use_case_str=use_case,
        )


# =============================================================================
# AttnProblem — natten attention (FMHA)
# =============================================================================


@dataclass(kw_only=True)
class AttnProblem(Problem):
    seqlen_q: int = 0
    seqlen_kv: int = 0
    is_causal: Any = False  # bool for AttnProblem; NAProblem shadows with tuple
    seqlens_q: Optional[List[int]] = None
    seqlens_kv: Optional[List[int]] = None
    heads_last: bool = True  # SdpaProblem flips this

    @classmethod
    def _from_args(cls, args) -> "AttnProblem":
        dtype = DTYPE_MAP[args.dtype]
        seqlen_kv = args.seqlen_kv if args.seqlen_kv is not None else args.seqlen
        varlen = getattr(args, "varlen", False)
        seqlens_q = getattr(args, "seqlens", None) if varlen else None
        seqlens_kv = getattr(args, "seqlens_kv", None) if varlen else None
        if varlen:
            if seqlens_q is None or seqlens_kv is None:
                raise ValueError(
                    "--seqlens and --seqlens-kv are both required when --varlen is set."
                )
            if len(seqlens_q) != len(seqlens_kv):
                raise ValueError(
                    "--seqlens and --seqlens-kv must have the same length "
                    f"(got {len(seqlens_q)} and {len(seqlens_kv)})."
                )
        return cls(
            batch_size=args.batch_size,
            heads=args.heads,
            heads_kv=args.heads_kv,
            dim=args.dim,
            dim_value=args.dim_value,
            dtype=dtype,
            bwd=args.bwd,
            perf=PerfConfig.from_args(args),
            seqlen_q=args.seqlen,
            seqlen_kv=seqlen_kv,
            is_causal=getattr(args, "is_causal", False),
            seqlens_q=seqlens_q,
            seqlens_kv=seqlens_kv,
        )

    @property
    def is_varlen(self) -> bool:
        return self.seqlens_q is not None

    # ---- Shapes / display ----
    def get_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self.is_varlen:
            assert self.seqlens_q is not None
            total_q = sum(self.seqlens_q)
            total_kv = sum(self.seqlens_kv) if self.seqlens_kv is not None else total_q
            b = 1
        else:
            total_q = self.seqlen_q
            total_kv = self.seqlen_kv
            b = self.batch_size

        if self.heads_last:
            return {
                "q": (b, total_q, self.heads, self.dim),
                "k": (b, total_kv, self.heads_kv, self.dim),
                "v": (b, total_kv, self.heads_kv, self.dim_value),
                "d_out": (b, total_q, self.heads, self.dim_value),
            }
        return {
            "q": (b, self.heads, total_q, self.dim),
            "k": (b, self.heads_kv, total_kv, self.dim),
            "v": (b, self.heads_kv, total_kv, self.dim_value),
            "d_out": (b, self.heads, total_q, self.dim_value),
        }

    def make_varlen_params(self, device: torch.device) -> dict:
        assert self.is_varlen and self.seqlens_q is not None
        seqlens_kv = self.seqlens_kv if self.seqlens_kv is not None else self.seqlens_q

        cum_q = [0]
        for s in self.seqlens_q:
            cum_q.append(cum_q[-1] + s)
        cum_kv = [0]
        for s in seqlens_kv:
            cum_kv.append(cum_kv[-1] + s)

        return {
            "cumulative_seqlen_Q": torch.tensor(
                cum_q, dtype=torch.int32, device=device
            ),
            "cumulative_seqlen_KV": torch.tensor(
                cum_kv, dtype=torch.int32, device=device
            ),
            "max_seqlen_Q": max(self.seqlens_q),
            "max_seqlen_KV": max(seqlens_kv),
        }

    def __str__(self) -> str:
        segments: List[str] = [
            f"batch={self.batch_size}",
            f"heads={self.heads}",
            f"heads_kv={self.heads_kv}",
            f"dim={self.dim}",
            f"dim_value={self.dim_value}",
            f"seqlen_q={self.seqlen_q}",
            f"seqlen_kv={self.seqlen_kv}",
            f"is_causal={self.is_causal}",
        ]
        if self.is_varlen:
            segments.append("varlen=True")
        segments.append(f"dtype={DTYPE_SHORT.get(self.dtype, str(self.dtype))}")
        return format_section("Problem:", segments)

    # ---- Op ----
    def _run_op(self, tensors: Dict[str, Tensor]) -> None:
        from nattenprof.ops import run_attn

        p = self.perf
        run_attn(
            tensors=tensors,
            problem=self,
            backend=p.fmha_backend,
            q_tile_size=p.q_tile_size,
            kv_tile_size=p.kv_tile_size,
            backward_q_tile_size=p.backward_q_tile_size,
            backward_kv_tile_size=p.backward_kv_tile_size,
            run_persistent_kernel=p.is_persistent,
            kernel_schedule=p.kernel_schedule,
            torch_compile=p.torch_compile,
            disable_backward=not self.bwd,
        )

    # ---- Check / dry-run / optimize ----
    def check_config(self) -> None:
        p = self.perf
        p.validate_attn_mode()
        p.normalize_schedule()
        q = torch.empty(self.get_tensor_shapes()["q"], dtype=self.dtype, device="cuda")
        p.resolve_fmha(q, is_causal=self.is_causal, is_varlen=self.is_varlen)

    def dry_run(self, max_configs: int) -> None:
        from nattenprof.dry_run import _dry_run_attn

        _dry_run_attn(self, max_configs=max_configs)

    def optimize(self, settings) -> None:
        from nattenprof.dry_run import _optimize_attn

        _optimize_attn(self, settings=settings)

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "seqlen_q": self.seqlen_q,
            "seqlen_kv": self.seqlen_kv,
            "batch_size": self.batch_size,
            "heads": self.heads,
            "heads_kv": self.heads_kv,
            "dim": self.dim,
            "dim_value": self.dim_value,
            "dtype": str(self.dtype),
            "is_causal": self.is_causal,
            "varlen": self.is_varlen,
            "fmha_backend": self.perf.fmha_backend,
        }

    def _operation_name(self) -> str:
        return "attn"


# =============================================================================
# SdpaProblem — torch SDPA (same data as AttnProblem; heads-first layout)
# =============================================================================


@dataclass(kw_only=True)
class SdpaProblem(AttnProblem):
    def __post_init__(self):
        # torch SDPA uses heads-first layout
        self.heads_last = False

    @classmethod
    def _from_args(cls, args) -> "SdpaProblem":
        dtype = DTYPE_MAP[args.dtype]
        seqlen_kv = args.seqlen_kv if args.seqlen_kv is not None else args.seqlen
        return cls(
            batch_size=args.batch_size,
            heads=args.heads,
            heads_kv=args.heads_kv,
            dim=args.dim,
            dim_value=args.dim_value,
            dtype=dtype,
            bwd=args.bwd,
            perf=PerfConfig.from_args(args),
            seqlen_q=args.seqlen,
            seqlen_kv=seqlen_kv,
            is_causal=getattr(args, "is_causal", False),
        )

    def _run_op(self, tensors: Dict[str, Tensor]) -> None:
        from nattenprof.ops import run_sdpa

        run_sdpa(
            tensors=tensors,
            backend=self.perf.fmha_backend,
            is_causal=self.is_causal,
            disable_backward=not self.bwd,
        )

    def check_config(self) -> None:
        p = self.perf
        p.validate_sdpa_mode()
        p.resolve_sdpa_default()

    def _config_dict(self) -> Dict[str, Any]:
        # SDPA doesn't support varlen — strip the inherited field after asserting it's unused.
        assert not self.is_varlen, "SDPA doesn't support varlen"
        out = super()._config_dict()
        del out["varlen"]
        return out

    def _operation_name(self) -> str:
        return "sdpa"


# =============================================================================
# NAProblem — natten neighborhood attention (FNA)
# =============================================================================


@dataclass(kw_only=True)
class NAProblem(AttnProblem):
    input_size: Tuple[int, ...] = field(default_factory=tuple)
    window_size: Tuple[int, ...] = field(default_factory=tuple)
    stride: Tuple[int, ...] = field(default_factory=tuple)
    dilation: Tuple[int, ...] = field(default_factory=tuple)
    additional_kv_length: int = 0
    # is_causal shadows parent bool with Tuple[bool, ...]

    @classmethod
    def _from_args(cls, args) -> "NAProblem":
        from natten.utils.checks import (
            check_all_args,
            check_input_size_arg,
            check_kernel_size_arg,
        )

        dtype = DTYPE_MAP[args.dtype]
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
        return cls(
            batch_size=args.batch_size,
            heads=args.heads,
            heads_kv=args.heads_kv,
            dim=args.dim,
            dim_value=args.dim_value,
            dtype=dtype,
            bwd=args.bwd,
            perf=PerfConfig.from_args(args),
            input_size=input_size,
            window_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=causal,
            additional_kv_length=args.add_kv,
        )

    @property
    def na_dim(self) -> int:
        return len(self.input_size)

    def is_self_attn(self) -> bool:
        """Matches natten.utils.checks.is_self_attention."""
        if not all(w == x for x, w in zip(self.input_size, self.window_size)):
            return False
        if self.na_dim > 1 and any(self.is_causal):
            return False
        return True

    def get_tensor_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Heads-last layout. NA kernels don't support heads-first."""
        shapes: Dict[str, Tuple[int, ...]] = {
            "q": (self.batch_size, *self.input_size, self.heads, self.dim),
            "k": (self.batch_size, *self.input_size, self.heads_kv, self.dim),
            "v": (self.batch_size, *self.input_size, self.heads_kv, self.dim_value),
            "d_out": (self.batch_size, *self.input_size, self.heads, self.dim_value),
        }
        if self.additional_kv_length > 0:
            shapes["add_k"] = (
                self.batch_size,
                self.additional_kv_length,
                self.heads_kv,
                self.dim,
            )
            shapes["add_v"] = (
                self.batch_size,
                self.additional_kv_length,
                self.heads_kv,
                self.dim_value,
            )
        return shapes

    def __str__(self) -> str:
        segments: List[str] = [
            f"batch={self.batch_size}",
            f"heads={self.heads}",
            f"heads_kv={self.heads_kv}",
            f"dim={self.dim}",
            f"dim_value={self.dim_value}",
            f"input_size={self.input_size}",
            f"window_size={self.window_size}",
            f"stride={self.stride}",
            f"dilation={self.dilation}",
            f"is_causal={self.is_causal}",
            f"dtype={DTYPE_SHORT.get(self.dtype, str(self.dtype))}",
        ]
        return format_section("Problem:", segments)

    def _run_op(self, tensors: Dict[str, Tensor]) -> None:
        from nattenprof.ops import run_na

        p = self.perf
        run_na(
            tensors=tensors,
            problem=self,
            backend=p.fna_backend,
            fmha_backend=p.fmha_backend,
            q_tile_shape=p.q_tile_shape,
            kv_tile_shape=p.kv_tile_shape,
            backward_q_tile_shape=p.backward_q_tile_shape,
            backward_kv_tile_shape=p.backward_kv_tile_shape,
            q_tile_size=p.q_tile_size,
            kv_tile_size=p.kv_tile_size,
            backward_q_tile_size=p.backward_q_tile_size,
            backward_kv_tile_size=p.backward_kv_tile_size,
            run_persistent_kernel=p.is_persistent,
            kernel_schedule=p.kernel_schedule,
            torch_compile=p.torch_compile,
            disable_backward=not self.bwd,
        )

    def check_config(self) -> None:
        p = self.perf
        p.normalize_schedule()
        q = torch.empty(self.get_tensor_shapes()["q"], dtype=self.dtype, device="cuda")
        # FNA resolution: NA-native path, always.
        p.resolve_fna(q, dilation=self.dilation)
        # FMHA resolution: self-attn runs full validation; non-self-attn only
        # needs the backend resolved (tiles come from FNA shapes).
        q_flat = q.flatten(1, self.na_dim)
        p.resolve_fmha(
            q_flat,
            is_causal=False,
            is_varlen=False,
            validate_tiles=self.is_self_attn(),
        )
        if not self.is_self_attn():
            p.derive_sizes_from_shapes()

    def dry_run(self, max_configs: int) -> None:
        from nattenprof.dry_run import _dry_run_na

        _dry_run_na(self, max_configs=max_configs)

    def optimize(self, settings) -> None:
        from nattenprof.dry_run import _optimize_na

        _optimize_na(self, settings=settings)

    def _config_dict(self) -> Dict[str, Any]:
        # NA doesn't use flat seqlen or varlen — assert the inherited fields are unused.
        assert (
            self.seqlen_q == 0 and self.seqlen_kv == 0
        ), "NAProblem uses input_size, not seqlen_q/seqlen_kv"
        assert not self.is_varlen, "NAProblem doesn't support varlen"

        out = super()._config_dict()
        for k in ("seqlen_q", "seqlen_kv", "varlen"):
            del out[k]

        out["input_size"] = self.input_size
        out["window_size"] = self.window_size
        out["stride"] = self.stride
        out["dilation"] = self.dilation

        # NA also emits fna_backend (parent already emits fmha_backend).
        out["fna_backend"] = self.perf.fna_backend

        return out

    def _operation_name(self) -> str:
        return "na"
