# Session: Port deterministic mode to Blackwell FMHA backward

## Task
Port deterministic backward mode from fbgemm_fmha/ to NATTEN's Blackwell FMHA backward kernel.
Expose via `torch.are_deterministic_algorithms_enabled()` in Python.

## Status: COMPLETE (all code changes done, not built/tested yet)

## What was done

### 10 files modified:

#### 1. Kernel: `csrc/include/natten/cuda/fmha_blackwell/kernel/sm100_fmha_bwd_kernel_tma_warpspecialized.hpp`
- User made these changes directly
- Added `bool IsDeterministic` template param (non-defaulted)
- Added `int* ptr_dq_semaphore = nullptr` to `MainloopArguments`
- Added `compute_expected_turn()` method
- Added `cutlass::GenericBarrier` sync in `reduce()`: `wait_eq` before dQ TMA store, `arrive_inc` after
- Added `#include "cutlass/barrier.h"`

#### 2. Device: `csrc/include/natten/cuda/fmha_blackwell/device/fmha_bwd_sm100.hpp`
- Added `bool IsDeterministic = false` template param to `FmhaBwdSm100`
- Added `int* ptr_dq_semaphore = nullptr` to `Arguments` (between `softmax_scale` and `hw_info`)
- Passed `IsDeterministic` to `Sm100FmhaBwdKernelTmaWarpSpecialized`
- Passed `args.ptr_dq_semaphore` through `to_bwd_arguments` into kernel MainloopArguments

#### 3. KernelBackward: `csrc/include/natten/cuda/fmha_blackwell/fmha_backward.cuh`
- Added `bool kIsDeterministic = false` template param
- Forwarded to `FmhaBwdSm100`
- Added `int* ptr_dq_semaphore` param to `initialize()`
- Passed it into `Arguments` between `attn_scale` and `hw_info`

#### 4. Autogen: `scripts/autogen_blackwell_fmha_bwd.py`
- Added `bool deterministic` to `KERNEL_DECL_TEMPLATE` and `KERNEL_IMPL_TEMPLATE`
- Impl template wraps all kernel logic in `auto run = [&](auto kIsDeterministic) { ... };`
- Uses `constexpr bool IsDeterministic = decltype(kIsDeterministic)::value;` to get compile-time bool
- All `KernelBackward<>` instantiations parameterized on `IsDeterministic`
- Semaphore allocated as `torch::zeros({num_q_blocks, batch_size, heads_q}, int32)` when deterministic
- Runtime dispatch: `if (deterministic) { run(std::true_type{}); } else { run(std::false_type{}); }`

#### 5. C++ header: `csrc/include/natten/blackwell_fmha.h`
- Added `bool deterministic` as last param

#### 6. C++ impl: `csrc/src/blackwell_fmha.cu`
- Added `bool deterministic` param
- Removed `TORCH_CHECK(not deterministicAlgorithms(), ...)` that blocked deterministic mode
- Passed `deterministic` to `DISPATCH_BLACKWELL_FMHA_BACKWARD` args

#### 7. Python torch_wrappers: `src/natten/_libnatten/torch_wrappers.py`
- Added `deterministic: bool` to `blackwell_fmha_backward_torch_op` and `blackwell_fmha_backward_torch_fake_op`
- Passed through to `blackwell_fmha_backward_cxx`

#### 8. Python backend: `src/natten/backends/blackwell_fmha.py`
- Removed `RuntimeError` that blocked deterministic mode
- Passes `torch.are_deterministic_algorithms_enabled()` as last arg to `blackwell_fmha_backward`

#### 9. Test: `tests/test_fmha.py`
- Added `test_cutlass_blackwell_fmha_determinism`
- 6 problem sizes (various batch/heads/GQA/dim/seqlen combos)
- float16 + bfloat16, causal + non-causal
- Runs forward+backward twice with `torch.use_deterministic_algorithms(True)`
- Asserts out, dq, dk, dv match at atol=1e-6, rtol=0 (unrolled per tensor)

#### 10. Test: `tests/test_fmha_varlen.py`
- Added `test_cutlass_blackwell_varlen_fmha_determinism`
- 4 varlen problem sizes
- Same pattern: 2 dtypes x 2 causal x 2 runs, assert all 4 tensors match

## Key design decisions
- `IsDeterministic` is a compile-time template parameter (matching fbgemm approach)
- Runtime bool converted to compile-time via `std::true_type`/`std::false_type` lambda dispatch
- Semaphore tensor shape: `{ceil(seqlen_q / kBlockM), batch, heads_q}` as int32
- For varlen, uses `max_seqlen_Q` instead of total seqlen for semaphore sizing
- `compute_expected_turn` simplified vs fbgemm: no window_size (NATTEN doesn't use windowed FMHA)
- Non-deterministic path has zero overhead (nullptr semaphore, `if constexpr` compiled out)

## Notes
- pybind11 binding in `csrc/natten.cpp` auto-infers new param from C++ signature (no change needed)
- Dispatch macros in autogen use `__VA_ARGS__` pass-through (no changes needed)
- Updated stale comments in test `_reset_everything()` ("Hopper and Blackwell" -> "Hopper")
