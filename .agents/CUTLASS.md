# CUTLASS Reference (NATTEN-relevant)

Docs: `third_party/cutlass/media/docs/cpp/` (NOT `third_party/cutlass/docs/`).
Headers: `third_party/cutlass/include/{cutlass,cute}/`.

## CuTe (CUTLASS 3.x foundation)

### Layout
A `Layout` = (`Shape`, `Stride`) maps coordinates to indices / offsets. Both Shape and Stride are
`IntTuple`s (recursive tuples of integers). Integers can be static (`cute::C<N>`, aliases `_1`, `_2`, ...)
or dynamic (`int`). Static integers enable compile-time optimizations and correctness checks.

Key operations: `rank`, `size`, `shape`, `stride`, `get<I>`. Layout algebra: composition, division
(tiling), slicing, flattening.

### Tensor
`Tensor` = Engine (random-access iterator) + `Layout`. The engine can be a pointer (gmem/smem),
register array, or implicit iterator (e.g. counting iterator for TMA coordinate tensors).

### Swizzle
`Swizzle<BBits, MBase, SShift>` — bitwise XOR permutation of memory offsets to reduce shared memory
bank conflicts. Applied to smem layouts via `composition(swizzle, layout)`.

### MMA Atom → TiledMMA
- **Operation struct**: wraps a single PTX MMA instruction (minimal deps, no CuTe types).
  Named like `SM90_64x128x16_F16F16F16F16_TN` (arch, dims, types, transpose).
- **MMA_Traits**: meta-info — `ValTypeA/B/C/D`, `Shape_MNK`, thread-to-data layouts (`ThrID`,
  `ALayout`, `BLayout`, `CLayout` as `(thread,value) → (M,N,K)` mappings).
- **TiledMMA**: tiles atoms across threads/data. Used with `cute::gemm(tiled_mma, a, b, c)`.

Atom granularity varies by arch: thread (FMA), quadpair (Volta), warp (Ampere), warpgroup (Hopper).

### Copy Atom → TiledCopy
Same structure as MMA but for data movement. Wraps `cp.async`, TMA, or plain loads/stores.
`TiledCopy` tiles across threads. Used with `cute::copy(tiled_copy, src, dst)`.

## CUTLASS 3.x GEMM Hierarchy

Five levels (top-down):

1. **Device** (`GemmUniversalAdapter`): host-side handle, launches kernel.
2. **Kernel** (`GemmUniversal`): stateless GPU kernel, schedules collectives over problem tiles.
   Persistent kernels loop over tiles via a tile scheduler; non-persistent use grid launch.
3. **Collective** (`CollectiveMma` + epilogue): the k-tile mainloop. Largest thread group
   cooperating via hardware (TMA, barriers, etc.). This is the main extension point.
4. **Tiled MMA/Copy**: `cute::gemm()` and `cute::copy()` with `TiledMma`/`TiledCopy` instances.
   Fully unrolled static loops.
5. **Atom**: raw PTX instruction wrappers.

NATTEN implements custom collectives (level 3) for both Hopper and Blackwell, and custom
threadblock-level kernels (CUTLASS 2.x style) for SM50–SM80.

### CollectiveMma template params
`DispatchPolicy`, `TileShape`, `ElementA/B`, `StrideA/B` (rank-3: `[outer, inner, batch]`),
`TiledMma`, `GmemTiledCopyA/B`, `SmemLayoutAtomA/B`, `SmemCopyAtomA/B`, `TransformA/B`.

### Dispatch Policies (tag-based)
Control which mainloop specialization is used. Example:
```cpp
MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>
```
Kernel schedules: `KernelTmaWarpSpecialized`, `KernelTmaWarpSpecializedPingpong`,
`KernelTmaWarpSpecializedCooperative` (Hopper); `KernelCpAsyncWarpSpecialized*` (Ampere).

### CollectiveBuilder
Simplified interface: takes `ArchTag`, `OpClass`, element types, alignments, `TileShape_MNK`,
`ClusterShape_MNK`, `StageCount`, `KernelSchedule` → produces a `CollectiveOp`.

### Pipeline
Producer-consumer async pipeline for hiding gmem latency.
- `PipelineAsync<NumStages>`: manages circular buffer of stages with barriers.
- API: `producer_acquire` (blocking), `producer_commit` (non-blocking),
  `consumer_wait` (blocking), `consumer_release` (non-blocking).
- `make_producer_start_state` for initial empty pipeline.
- TMA-based producers auto-update barriers (`producer_commit` becomes no-op).

## Hopper (SM90) specifics

### TMA (Tensor Memory Accelerator)
Bulk multidimensional copy between gmem and smem via hardware. One instruction copies an entire tile.
- **TMA descriptor**: created on host, holds base pointer, shape, strides, swizzle, OOB behavior.
  Shared across all thread blocks.
- Kernel receives: descriptor pointer, smem pointer, coordinates (not gmem pointer).
- CuTe represents TMA tensors with `ArithmeticTupleIterator` (implicit coordinate tensor) so
  tiling/slicing TMA coordinate tensors works identically to data tensors.

### WGMMA (Warpgroup MMA)
Warpgroup = 4 warps (128 threads). Operates on data in shared memory or registers.
Atoms: `SM90_64xNx16_*` family in `cute/arch/mma_sm90_gmma.hpp`.

### Thread Block Clusters
New hierarchy level: group of thread blocks that can share smem via TMA multicast.
`ClusterShape` in dispatch policies controls this.

## Blackwell (SM100) specifics

### tcgen05.mma
New MMA instructions, 2–4x Hopper throughput. Operate on tensor memory (TMEM).
Support all legacy types (tf32, f16, bf16, i8) plus narrow precision (f8/f6/f4).

### TMEM (Tensor Memory)
Dedicated on-chip memory for tcgen05.mma accumulator and operand storage.
Allocated via `cute::arch::tmem_alloc_sm100`. NATTEN uses TMEM allocation and SIMD operations
(`cute/arch/tmem_allocator_sm100.hpp`, `cute/arch/simd_sm100.hpp`).

### cta_group
Blackwell collectives can span 1 or 2 CTAs cooperating on a single MMA
(`cta_group::1` or `cta_group::2`).

## CUTLASS 2.x (SM50–SM80, brief)

NATTEN's `cutlass-fmha` and `cutlass-fna` targets use the 2.x API for SM50/SM70/SM75/SM80.

### Key abstractions
- **Threadblock-level MMA** (`cutlass::gemm::threadblock`): `MmaPipelined` (2-stage, SM70),
  `MmaMultistage` (N-stage, SM80). NATTEN extends these with custom variants (`custom_mma_*.h`).
- **Warp-level MMA** (`cutlass::gemm::warp`): warp tile iterators, MMA tensor op instructions.
- **Epilogue** (`cutlass::epilogue::threadblock`): predicated tile iterators, linear combination.
- **Iterators** (`cutlass::transform::threadblock`): `PredicatedTileIterator`,
  `PredicatedTileAccessIterator` — handle OOB access with predicates.
- **Layouts** (`cutlass::layout`): `RowMajor`, `ColumnMajor`, `PitchLinear`.

2.x uses bespoke named types per arch rather than tag dispatch. Thread-to-data mapping is implicit
in iterator logic (vs. explicit CuTe layouts in 3.x).

## Include structure

```
cutlass/
  arch/          # Arch-specific: barriers, memory ops, MMA instructions
  gemm/
    collective/  # 3.x CollectiveMma specializations per arch
    kernel/      # Kernel-level: tile schedulers, GemmUniversal
    device/      # Host adapter (GemmUniversalAdapter)
    threadblock/ # 2.x threadblock MMA (MmaPipelined, MmaMultistage)
    warp/        # 2.x warp-level MMA tile iterators
  epilogue/
    collective/  # 3.x epilogues
    threadblock/ # 2.x epilogues
    thread/      # Element-wise ops (LinearCombination, activations)
  pipeline/      # PipelineAsync (SM90), SM100 pipeline
  transform/
    threadblock/ # 2.x tile iterators (PredicatedTileIterator, etc.)
  layout/        # RowMajor, ColumnMajor, PitchLinear, etc.

cute/
  layout.hpp           # Core Layout abstraction
  tensor.hpp           # Tensor = Engine + Layout
  swizzle.hpp          # Swizzle functor
  int_tuple.hpp        # IntTuple operations
  algorithm/           # copy, gemm, fill, clear, axpby
  arch/                # PTX wrappers: mma_sm{70,75,80,90,100}.hpp, copy_sm*.hpp,
                       #   cluster_sm90.hpp, tmem_allocator_sm100.hpp, simd_sm100.hpp
  atom/                # MMA_Traits and Copy_Traits per arch
    mma_atom.hpp       # MMA_Atom combining Operation + Traits
    copy_atom.hpp      # Copy_Atom combining Operation + Traits
```

## Connections to NATTEN
Blackwell FMHA / FNA kernels are based on example 77 in cutlass
(`third_party/cutlass/examples/77_blackwell_fmha`).

Hopper FMHA/ FNA kernels are based on example 88 in cutlass
(`third_party/cutlass/examples/88_hopper_fmha`).

CUTLASS FMHA/ FNA kernels are based on example 41 in cutlass
(`third_party/cutlass/examples/41_fused_multi_head_attention`).
NOTE: the FNA variant does NOT use token permutation, so it requires many more changes to the
original FMHA kernel.
