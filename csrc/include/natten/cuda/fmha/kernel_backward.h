/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#include <natten/cuda/fmha/gemm_kernel_utils.h>
#include <natten/natten.h>

#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/threadblock/epilogue_smem_accumulator.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm80.h>
#include <cutlass/integer_subbyte.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/platform/platform.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/vector_iterator.h>

#include <natten/cuda/fmha/epilogue/epilogue_pipelined.h>
#include <natten/cuda/fmha/iterators/epilogue_predicated_tile_iterator.h>

#include <natten/cuda/fmha/gemm/custom_mma.h>
#include <natten/cuda/fmha/gemm/find_default_mma.h>
#include <natten/cuda/fmha/gemm/mma_accum_lambda_iterator.h>
#include <natten/cuda/fmha/gemm/mma_from_smem.h>
#include <natten/cuda/fmha/transform/tile_smem_loader.h>

#include <inttypes.h>

namespace natten {
namespace cuda {
namespace fmha {

using namespace gemm_kernel_utils;

namespace {

template <typename FragmentType, int32_t kNumThreads>
struct GmemTile {
  /*
    Helper functions to efficient store/load RF to gmem

    GEMM accumulators have a particular format on A100, and
    it takes some compute/shared-memory to rearrange them to
    a RowMajor or ColumnMajor format in global memory through
    an Epilogue. The same complexity goes for loading into RF.

    This class loads/stores RF as they are, and can be used for
    efficient accumulation across gemms for instance:

    ```
    GmemTile tile;
    for (int i = 0; i < N; ++i) {
      // ...

      Fragment accum;
      if (i == 0) {
        accum.clear();
      } else {
        tile.load(accum);
      }
      mma(accum, ...);
      if (i < N-1) {
        // Store for next GEMM
        tile.store(accum);
      } else {
        // Store in tensor (eg RowMajor)
        epilogue(accum);
      }

      // ...
    }
    ```
  */

  // 128bits per thread
  using AccessType = cutlass::Array<float, 4>;
  static constexpr int32_t kBytes = sizeof(AccessType);
  static constexpr int32_t kStride = kNumThreads * AccessType::kElements;
  static constexpr int32_t kNumIters =
      FragmentType::kElements / AccessType::kElements;
  static constexpr int32_t kElementsStored =
      kNumThreads * FragmentType::kElements;
  static_assert(
      FragmentType::kElements % AccessType::kElements == 0,
      "fragment not aligned on 128 bits");

  float* ptr;

  CUTLASS_DEVICE void load(FragmentType& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      cutlass::arch::global_load<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        fragment[i * AccessType::kElements + j] = sub_fragment[j];
      }
    }
  }

  CUTLASS_DEVICE void store(FragmentType const& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        sub_fragment[j] = fragment[i * AccessType::kElements + j];
      }
      cutlass::arch::global_store<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
    }
  }
};

struct AtomicLock {
  CUTLASS_DEVICE static void acquire(
      int32_t* lock,
      int set_val,
      int thread_id) {
    if (thread_id == 0) {
      while (atomicCAS(lock, 0 /*cmp*/, set_val /*setval*/) != set_val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(40);
#endif
      }
    }
    __syncthreads();
  }
  CUTLASS_DEVICE static void release(int32_t* lock, int thread_id) {
    if (thread_id == 0) {
      int status = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(status));
#else
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};

template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmBw() {
  bool is_half = !cutlass::platform::is_same<scalar_t, float>::value;
  if (Arch::kMinComputeCapability >= 80) {
    return is_half ? 12 : 8;
  }
  return 8;
}
} // namespace

template <
    // input/output type
    typename scalar_t_,
    // which arch we target (eg `cutlass::arch::Sm80`)
    typename ArchTag_,
    // run optimized kernel because memory accesses will be aligned
    bool kIsAligned_,
    // NOTE(alih): we force alignment on dim in NATTEN, so we
    // only need kPreload_=true instantiations.
    ////  when doing a GEMM, preload the next one (uses more shmem)
    //// bool kPreload_ = true,
    // block dimensions
    int kBlockSizeI_,
    int kBlockSizeJ_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    // assumes that:
    // (1) `num_queries % kBlockSizeI == 0`
    // (2) `num_keys % kBlockSizeJ == 0`
    bool kKeysQueriesAlignedToBlockSize_ = false>
struct AttentionBackwardKernel {
  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using output_accum_t = float;
  using lse_scalar_t = float;
  using accum_t = float;
  using ArchTag = ArchTag_;
  static constexpr bool kIsAligned = kIsAligned_;
  // static constexpr bool kPreload = kPreload_;
  static constexpr int kBlockSizeI = kBlockSizeI_;
  static constexpr int kBlockSizeJ = kBlockSizeJ_;
  static constexpr int kMaxK = kMaxK_;
  static constexpr bool kKeysQueriesAlignedToBlockSize =
      kKeysQueriesAlignedToBlockSize_;

  static constexpr int64_t kWarpSize = 32;

  // If this is true, we store and accumulate dK/dV in RF
  // rather than going back to gmem everytime
  static constexpr bool kIsHalf = cutlass::sizeof_bits<scalar_t>::value <= 16;
  static constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;

  static constexpr bool kPreload =
      kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF;
  static_assert(
      !kPreload ||
          (kIsHalf && ArchTag::kMinComputeCapability >= 80 && kOutputInRF),
      "preload MMA not supported");
  static constexpr bool kPrologueQK = kPreload;
  static constexpr bool kPrologueGV = kPreload;
  static constexpr bool kPrologueDOV = kPreload;
  static constexpr bool kPrologueGQ = kPreload;
  static constexpr bool kPrologueGK = kPreload;

  static constexpr int64_t kNumWarpsPerBlock =
      (kBlockSizeI * kBlockSizeJ) / (32 * 32);

  // Compute delta for the f16 kernels
  // TODO: Figure out why it's slower on the f32 kernels
  // (something due to RF pressure?)
  // TODO: Remove condition on `kOutputInRF` - this is needed to work
  // around a compiler bug on V100, not exactly sure why but I spent
  // too much time on this already. Reproducible with
  // (B, Mq, Mkv, K) = (1, 1, 1, 136) for instance
  // TODO (alih): this was buggy when I was doing FNA in 2024, so disabling for
  // safety
  // static constexpr bool kKernelComputesDelta =
  //    kIsHalf && (kOutputInRF || ArchTag::kMinComputeCapability != 70);
  static constexpr bool kKernelComputesDelta = false;

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSmBw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  using GemmType = DefaultGemmType<ArchTag, scalar_t>;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          typename GemmType::OpClass,
          ArchTag,
          scalar_t,
          scalar_t,
          scalar_t, // ElementC
          accum_t // ElementAccumulator
          >;
  static constexpr auto kOptimalAlignement = cutlass::platform::max(
      DefaultConfig::kAlignmentA,
      DefaultConfig::kAlignmentB);
  static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

  struct MatmulQK {
    /*
    attn_T = k_j @ q_i.transpose(-2, -1) # matmul
    attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
    -1)).exp() # epilogue

    with attn_T.shape = (kBlockSizeJ, kBlockSizeI)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::DefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator,
        false, // AccumulatorsInRowMajor = false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Operator::
                InstructionShape, // InstructionShape
            typename DefaultGemm::Mma::Operator::
                IteratorA, // RegularWarpIterator
            typename DefaultGemm::Mma::Policy // Policy
            >::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulQK::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            /*kApplyDropout*/ false>; // kScaleOperandA

    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;

    using ElementC = output_t;
    using ElementAccum = accum_t;

    // no-op output op - epilogue just stores result to global memory
    using BiasGradEpilogueOutputOp =
        typename cutlass::epilogue::thread::LinearCombination<
            ElementC,
            DefaultConfig::EpilogueOutputOp::kCount,
            typename DefaultConfig::EpilogueOutputOp::ElementAccumulator,
            typename DefaultConfig::EpilogueOutputOp::ElementCompute,
            cutlass::epilogue::thread::ScaleType::Nothing>;

    using DefaultGemm = typename cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        ElementC, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        ElementAccum, // ElementAccumulator
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        BiasGradEpilogueOutputOp, // EpilogueOutputOp
        void, // ThreadblockSwizzle (not used)
        /*// multiple preloads, and 3 stages push us over shared
        // memory capacity on A100.
        kPreload && (kBlockSizeI * kBlockSizeJ > 64 * 64)
            ? cutlass::const_min(2, DefaultConfig::kStages)
            : */
        DefaultConfig::kStages, // Stages
        false, // SplitKSerial
        typename GemmType::Operator,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using Mma = typename MakeCustomMma<typename DefaultGemm::Mma, kMaxK>::Mma;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        ElementAccum,
        kWarpSize>::Iterator;

    // epilogue used to write bias gradient, which is just the output of this
    // matmul with some operations applied to the fragment
    using BiasGradEpilogue = typename DefaultGemm::Epilogue;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename DefaultGemm::Mma::Operator::IteratorC,
        typename DefaultGemm::Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradQ {
    // grad_q <- tmp @ k_j
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  struct MatmulGradK {
    // grad_k <- tmp.transpose(-2, -1) @ q_i
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kBlockSizeI, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmemN =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulQK::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            false>; // kScaleOperandA
    using DefaultMmaFromSmemT =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kM, // kMaxK
            WarpIteratorA,
            false, // kScaleOperandA
            kPreload>; // kTransposeA
    using DefaultMmaFromSmem = typename cutlass::platform::conditional<
        DefaultMmaFromSmemT::kIsTransposedA,
        DefaultMmaFromSmemT,
        DefaultMmaFromSmemN>::type;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  static constexpr bool kEnableSplitKeys = true;

  static constexpr bool kNeedsAccumGradQ = kEnableSplitKeys ||
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  static constexpr bool kNeedsAccumGradK = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;
  static constexpr bool kNeedsAccumGradV = !kOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  struct GradQTempStorage {
    int32_t lock;
    int32_t counter;
    int32_t pad[2]; // pad to 128bits
    output_accum_t buffer[MatmulGradQ::AccumTileGmem::kElementsStored];
  };

  struct Params {
    // Input tensors
    scalar_t* query_ptr = nullptr; // [Mq, nH, K]
    scalar_t* key_ptr = nullptr; // [Mk, nH, K]
    scalar_t* value_ptr = nullptr; // [Mk, nH, Kv]
    lse_scalar_t* logsumexp_ptr = nullptr; // [Mq, nH]
    scalar_t* output_ptr = nullptr; // [Mq, nH, Kv]
    scalar_t* grad_output_ptr = nullptr; // [Mq, nH, Kv]
    accum_t* delta_ptr = nullptr; // [nH, Mq]

    // Output tensors
    output_t* grad_query_ptr = nullptr; //  [Mq, nH, K]
    output_t* grad_key_ptr = nullptr; //    [Mk, nH, K]
    output_t* grad_value_ptr = nullptr; //  [Mk, nH, Kv]

    // Accumulators
    output_accum_t* workspace = nullptr; // [Mq, Kq] + [Mkv, Kq] + [Mkv, Kv]
    output_accum_t* workspace_gv =
        nullptr; // (will be calculated by the kernel)
    GradQTempStorage* workspace_gq =
        nullptr; // (will be calculated by the kernel)

    // Scale
    accum_t scale = 1.0f;

    // Dimensions/strides
    int32_t head_dim = -1;
    int32_t head_dim_value = -1;
    int32_t num_queries = -1;
    int32_t num_keys = -1;
    int32_t num_heads = -1;

    int32_t q_strideM = -1;
    int32_t k_strideM = -1;
    int32_t v_strideM = -1;
    int32_t gO_strideM = -1;

    CUTLASS_HOST_DEVICE int32_t o_strideM() const {
      return head_dim_value * num_heads;
    }
    CUTLASS_HOST_DEVICE int32_t gQ_strideM() const {
      return num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int32_t gK_strideM() const {
      return num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int32_t gV_strideM() const {
      return num_heads * head_dim_value;
    }

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int64_t o_strideH = -1;
    int32_t q_strideH = -1;
    int32_t k_strideH = -1;
    int32_t v_strideH = -1;
    int64_t o_strideB = -1;
    int64_t q_strideB = -1;
    int64_t k_strideB = -1;
    int64_t v_strideB = -1;
    int64_t lse_strideB = -1;
    int64_t delta_strideB = -1;
    int32_t num_batches = -1;
    int16_t num_splits_key = 1; // We use `gridDim.x` inside kernel

    int64_t gO_strideB = 0;
    int64_t gQ_strideB = 0;
    int64_t gK_strideB = 0;
    int64_t gV_strideB = 0;
    int64_t gO_strideH = 0;
    int64_t gQ_strideH = 0;
    int64_t gK_strideH = 0;
    int64_t gV_strideH = 0;

    CUTLASS_HOST_DEVICE int16_t num_splits_key_device() const {
#ifdef __CUDA_ARCH__
      return kEnableSplitKeys ? gridDim.x : 1;
#else
      return num_splits_key; // for host-side tests
#endif
    }

    CUTLASS_HOST_DEVICE int16_t split_key_device() const {
#ifdef __CUDA_ARCH__
      return kEnableSplitKeys ? blockIdx.x : 0;
#else
      return 0; // for host-side tests
#endif
    }

    CUTLASS_DEVICE bool advance_to_block() {
      int64_t batch_id = blockIdx.z;
      int32_t head_id = blockIdx.y;

      if (kNeedsAccumGradQ || kNeedsAccumGradK || kNeedsAccumGradV) {
        assert(workspace_size() == 0 || workspace != nullptr);

        workspace += (batch_id * num_heads + head_id) * workspace_strideBH();
        workspace = warp_uniform(workspace);
        workspace_gv = workspace + workspace_elements_gk();
        workspace_gq =
            (GradQTempStorage*)(workspace_gv + workspace_elements_gv());
        if (kEnableSplitKeys) {
          workspace_gv += workspace_elements_gv() * split_key_device() /
              num_splits_key_device();
          workspace += workspace_elements_gk() * split_key_device() /
              num_splits_key_device();
        }
      } else {
        workspace = nullptr;
      }

      // Advance pointers that depend on the total concatenated
      // number of queries, as `num_queries` is modified in the block
      // below
      logsumexp_ptr += batch_id * lse_strideB + head_id;
      delta_ptr += batch_id * delta_strideB + head_id;

      query_ptr += batch_id * q_strideB + head_id * q_strideH;
      key_ptr += batch_id * k_strideB + head_id * k_strideH;
      value_ptr += batch_id * v_strideB + head_id * v_strideH;
      output_ptr += batch_id * o_strideB + head_id * o_strideH;
      grad_output_ptr += batch_id * gO_strideB + head_id * gO_strideH;

      grad_query_ptr += batch_id * gQ_strideB + head_id * gQ_strideH;
      grad_key_ptr += batch_id * gK_strideB + head_id * gK_strideH;
      grad_value_ptr += batch_id * gV_strideB + head_id * gV_strideH;

      // Some values are modified above
      // Signal to the compiler that they are the same in all threads
      // and can be stored in warp-uniform registers (Sm75+)
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);

      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      output_ptr = warp_uniform(output_ptr);
      grad_output_ptr = warp_uniform(grad_output_ptr);
      delta_ptr = warp_uniform(delta_ptr);

      grad_query_ptr = warp_uniform(grad_query_ptr);
      grad_key_ptr = warp_uniform(grad_key_ptr);
      grad_value_ptr = warp_uniform(grad_value_ptr);

      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(num_splits_key, num_heads, num_batches);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize * kNumWarpsPerBlock, 1, 1);
    }

    CUTLASS_HOST_DEVICE int64_t workspace_elements_gk() const {
      if constexpr (!kNeedsAccumGradK) {
        return 0;
      }
      return num_splits_key * kBlockSizeJ *
          align_up(head_dim, (int32_t)kBlockSizeI);
    }

    CUTLASS_HOST_DEVICE int64_t workspace_elements_gv() const {
      if constexpr (!kNeedsAccumGradV) {
        return 0;
      }
      return num_splits_key * kBlockSizeJ *
          align_up(head_dim_value, (int32_t)kBlockSizeI);
    }

    CUTLASS_HOST_DEVICE int64_t workspace_elements_gq() const {
      if constexpr (!kNeedsAccumGradQ) {
        return 0;
      }
      int num_blocks = ceil_div(num_queries, kBlockSizeI);
      int num_cols = ceil_div(head_dim, MatmulGradQ::ThreadblockShape::kN);
      return num_blocks * num_cols * sizeof(GradQTempStorage) /
          sizeof(output_accum_t);
    }
    CUTLASS_HOST_DEVICE int64_t workspace_strideBH() const {
      // Aligned on 128bits
      return align_up(
          workspace_elements_gk() + workspace_elements_gv() +
              workspace_elements_gq(),
          int64_t(4));
    }
    CUTLASS_HOST_DEVICE int64_t workspace_size() const {
      // Returns size of buffer we need to run this kernel
      return num_batches * num_heads * workspace_strideBH() * sizeof(float);
    }
    CUTLASS_HOST_DEVICE bool should_zero_workspace() const {
      return num_splits_key > 1;
    }
  };

  struct SharedStoragePrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
      cutlass::Array<accum_t, kBlockSizeI> lse_i; // logsumexp
      typename MatmulQK::Mma::SharedStorageA mm_qk_k;
    } persistent;
    union {
      struct {
        // part1 - after Q.K / dV / dO.V
        union {
          // 4. store Pij. it is needed:
          // - in dVj += (Pij.T * Zij) @ dOi
          // - in dSij = Pij * (dPij - Di)
          // 6. dVj += (Pij.T * Zij) @ dOi
          // 10. write to fragment
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };

        union {
          // 2. prologue for dVj
          // 6. workspace for dVj += (Pij.T * Zij) @ dOi
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          // 7. dVj epilogue
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };

        // 3. prologue for dPij_dropped
        // 8. used in dPij_dropped = dOi @ Vj.T
        typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
      } part1;

      struct {
        // part2 - dQ
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
        union {
          // store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
        };

      } part2;

      struct {
        // part3 - after last iteration on dQ's epilogue / dK
        union {
          typename MatmulQK::AccumulatorSharedStorage
              tmpT_shared_storage; // (from part1)
          typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        };
        typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
        typename MatmulGradQ::DefaultEpilogue::SharedStorage
            gradQ_epilogue_lastIter;

        typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
      } part3;

      struct {
        // part4 - after last iteration on dK's epilogue / preload next K.Q_t
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;

        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part4;
    };
    static void print_size() {
      // Field size
#define FSZ(f) int((sizeof(((SharedStoragePrologue*)0)->f)))

      printf("Total smem: %d bytes\n", int(sizeof(SharedStoragePrologue)));
      printf("  persistent: %db\n", FSZ(persistent));
      printf("    mm_qk_k: %db\n", FSZ(persistent.mm_qk_k));
      printf("  part1: %db\n", FSZ(part1));
      printf("    attn_shared_storage: %db\n", FSZ(part1.attn_shared_storage));
      printf("    zij: %db\n", FSZ(part1.zij));
      printf("    mm_gradV: %db\n", FSZ(part1.mm_gradV));
      printf("    gradV_epilogue: %db\n", FSZ(part1.gradV_epilogue));
      printf("    mm_doivj: %db\n", FSZ(part1.mm_doivj));
      printf("  part2: %db\n", FSZ(part2));
      printf("    tmpT_shared_storage: %db\n", FSZ(part2.tmpT_shared_storage));
      printf("    tmp_shared_storage: %db\n", FSZ(part2.tmp_shared_storage));
      printf("    mm_gradK: %db\n", FSZ(part2.mm_gradK));
      printf("    mm_gradQ: %db\n", FSZ(part2.mm_gradQ));
      printf("    gradB_epilogue: %db\n", FSZ(part2.gradB_epilogue));
      printf("    gradQ_epilogue: %db\n", FSZ(part2.gradQ_epilogue));
      printf("  part3: %db\n", FSZ(part3));
      printf("    tmpT_shared_storage: %db\n", FSZ(part3.tmpT_shared_storage));
      printf("  part4: %db\n", FSZ(part4));
      printf("    mm_qk_q: %db\n", FSZ(part4.mm_qk_q));
      printf(
          "    gradK_epilogue_final: %db\n", FSZ(part4.gradK_epilogue_final));
      printf(
          "    gradV_epilogue_final: %db\n", FSZ(part4.gradV_epilogue_final));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(persistent, mm_qk_k)
    FIELD(persistent, lse_i)
    FIELD(part1, attn_shared_storage)
    FIELD(part1, zij)
    FIELD(part1, mm_gradV)
    FIELD(part1, gradV_epilogue)
    FIELD(part1, mm_doivj)
    FIELD(part2, mm_gradK)
    FIELD(part2, mm_gradQ)
    FIELD(part2, gradB_epilogue)
    FIELD(part2, gradQ_epilogue)
    FIELD(part2, tmp_shared_storage)
    FIELD(part3, tmpT_shared_storage)
    FIELD(part3, gradQ_epilogue_lastIter)
    FIELD(part3, gradK_epilogue)
    FIELD(part4, mm_qk_q)
    FIELD(part4, gradK_epilogue_final)
    FIELD(part4, gradV_epilogue_final)
  };

  struct SharedStorageNoPrologue {
    struct {
      cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
    } persistent;
    union {
      struct {
        // part1 - Q.K matmul
        typename MatmulQK::Mma::SharedStorageA mm_qk_k;
        typename MatmulQK::Mma::SharedStorageB mm_qk_q;
        cutlass::Array<accum_t, kBlockSizeI> lse_i; // logsumexp
      } part1;

      struct {
        // part2 - compute gradV
        union {
          // 2. store Pij to shared memory. it is needed:
          // - in this step, where it is used in dVj += (Pij.T * Zij) @ dOi
          // - in next step where it is used in dSij = Pij * (dPij - Di)
          typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
        };

        union {
          typename MatmulGradV::Mma::SharedStorage mm_gradV;
          typename MatmulGradV::DefaultEpilogue::SharedStorage gradV_epilogue;
        };
      } part2;

      struct {
        // part3 - DO.V matmul
        union {
          // first compute dPij = (dOi @ Vj.T) * Zij
          // and dSij = Pij * (dPij - Di)
          struct {
            // (from part2) - Pij for computing dSij = Pij * (dPij - Di)
            typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
            // matmul to compute dOiVj
            typename MatmulDOIVJ::Mma::SharedStorage mm_doivj;
          };
          // then store dB = dSij to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
        };
      } part3;

      struct {
        // part4 - compute gradQ
        typename MatmulQK::AccumulatorSharedStorage
            tmpT_shared_storage; // (from part2)
        typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        union {
          typename MatmulGradQ::Mma::SharedStorage mm_gradQ;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage gradQ_epilogue;
          typename MatmulGradQ::DefaultEpilogue::SharedStorage
              gradQ_epilogue_lastIter;
        };
      } part4;

      struct {
        // part5 - compute gradK
        typename MatmulQK::AccumulatorSharedStorage
            tmpT_shared_storage; // (from part2)
        typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
        union {
          typename MatmulGradK::Mma::SharedStorage mm_gradK;
          typename MatmulGradK::DefaultEpilogue::SharedStorage gradK_epilogue;
        };
      } part5;

      struct {
        // part6 - store RF accumulated into gmem
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } part6;
    };
    static void print_size() {
#define FIELD_SIZEOF(f) int((sizeof(((SharedStorageNoPrologue*)0)->f)))
      printf("Total smem: %d bytes\n", int(sizeof(SharedStorageNoPrologue)));
      printf("  persistent: %db\n", FIELD_SIZEOF(persistent));
      printf("  part1: %db\n", FIELD_SIZEOF(part1));
      printf("  part2: %db\n", FIELD_SIZEOF(part2));
      printf("  part3: %db\n", FIELD_SIZEOF(part3));
      printf("  part4: %db\n", FIELD_SIZEOF(part4));
      printf("  part5: %db\n", FIELD_SIZEOF(part5));
      printf("  part6: %db\n", FIELD_SIZEOF(part6));
    }
// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE auto& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(persistent, di)
    FIELD(part1, lse_i)
    FIELD(part1, mm_qk_k)
    FIELD(part1, mm_qk_q)
    FIELD(part2, attn_shared_storage)
    FIELD(part2, zij)
    FIELD(part2, mm_gradV)
    FIELD(part2, gradV_epilogue)
    FIELD(part3, mm_doivj)
    FIELD(part3, gradB_epilogue)
    FIELD(part4, tmpT_shared_storage)
    FIELD(part4, tmp_shared_storage)
    FIELD(part4, mm_gradQ)
    FIELD(part4, gradQ_epilogue)
    FIELD(part4, gradQ_epilogue_lastIter)
    FIELD(part5, mm_gradK)
    FIELD(part5, gradK_epilogue)
    FIELD(part6, gradK_epilogue_final)
    FIELD(part6, gradV_epilogue_final)
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kPreload,
      SharedStoragePrologue,
      SharedStorageNoPrologue>::type;

  struct OutputFragments {
    typename MatmulGradV::Mma::FragmentC gradV;
    typename MatmulGradK::Mma::FragmentC gradK;

    CUTLASS_DEVICE void clear() {
      gradV.clear();
      gradK.clear();
    }
  };

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.key_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.value_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.grad_output_ptr, kMinimumAlignment);
    NATTEN_CHECK(
        p.num_heads <= 1 || p.q_strideH % kMinimumAlignment == 0,
        "query is not correctly aligned (strideH)");
    NATTEN_CHECK(
        p.num_heads <= 1 || p.k_strideH % kMinimumAlignment == 0,
        "key is not correctly aligned (strideH)");
    NATTEN_CHECK(
        p.num_heads <= 1 || p.v_strideH % kMinimumAlignment == 0,
        "value is not correctly aligned (strideH)");
    NATTEN_CHECK(
        p.num_batches <= 1 || p.q_strideB % kMinimumAlignment == 0,
        "query is not correctly aligned (strideB)");
    NATTEN_CHECK(
        p.num_batches <= 1 || p.k_strideB % kMinimumAlignment == 0,
        "key is not correctly aligned (strideB)");
    NATTEN_CHECK(
        p.num_batches <= 1 || p.v_strideB % kMinimumAlignment == 0,
        "value is not correctly aligned (strideB)");
    NATTEN_CHECK(
        p.q_strideM % kMinimumAlignment == 0,
        "query is not correctly aligned (strideM)");
    NATTEN_CHECK(
        p.k_strideM % kMinimumAlignment == 0,
        "key is not correctly aligned (strideM)");
    NATTEN_CHECK(
        p.v_strideM % kMinimumAlignment == 0,
        "value is not correctly aligned (strideM)");
    NATTEN_CHECK(p.head_dim > 0, "Invalid value for `head_dim`");
    NATTEN_CHECK(p.head_dim_value > 0, "Invalid value for `head_dim_value`");
    NATTEN_CHECK(p.num_queries > 0, "Invalid value for `num_queries`");
    NATTEN_CHECK(p.num_keys > 0, "Invalid value for `num_keys`");
    NATTEN_CHECK(p.num_heads > 0, "Invalid value for `num_heads`");
    NATTEN_CHECK(p.num_batches > 0, "Invalid value for `num_batches`");
    NATTEN_CHECK(p.head_dim <= kMaxK, "kMaxK: Expected `head_dim < kMaxK`");
    NATTEN_CHECK(
        p.head_dim_value <= kMaxK, "kMaxK: Expected `head_dim_value < kMaxK`");
    if (kKeysQueriesAlignedToBlockSize) {
      NATTEN_CHECK(
          p.num_queries % kBlockSizeI == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
      NATTEN_CHECK(
          p.num_keys % kBlockSizeJ == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
    }
    NATTEN_CHECK(
        kEnableSplitKeys || p.num_splits_key == 1, "SplitKeys is disabled");
    NATTEN_CHECK(
        p.num_splits_key > 0, "Invalid `num_splits_key` (expected >0)");
    NATTEN_CHECK(
        p.num_splits_key <= cutlass::ceil_div(p.num_keys, kBlockSizeJ),
        "Invalid `num_splits_key` ("
            << p.num_splits_key
            << ") - too large for `num_keys` = " << p.num_keys);
    return true;
  }

  static CUTLASS_DEVICE void attention_kernel(Params p) {
    extern __shared__ char smem_buffer[];
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);

    uint16_t thread_id = threadIdx.x;
    uint8_t warp_id = warp_uniform(thread_id / 32);
    uint8_t lane_id = thread_id % 32;

    int32_t key_start = p.split_key_device() * kBlockSizeJ;
    if (key_start >= p.num_keys) {
      return;
    }
    if (kPrologueQK) {
      int32_t query_start = getQueryStart(p, key_start);
      prologueQkNextIteration<true>(
          shared_storage, p, query_start, key_start, warp_id, lane_id);
    }

    // Computes (dO*out).sum(-1) and writes it to `p.delta_ptr`
    if constexpr (kKernelComputesDelta) {
      constexpr int kOptimalElements =
          128 / cutlass::sizeof_bits<scalar_t>::value;
      if (p.head_dim_value % kOptimalElements == 0) {
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<kOptimalElements>(p, query_start, warp_id, lane_id);
        }
      } else {
        for (int query_start = 0; query_start < p.num_queries;
             query_start += kBlockSizeI) {
          computeDelta<1>(p, query_start, warp_id, lane_id);
        }
      }
      __syncthreads();
    }

    OutputFragments output_frags;

    CUTLASS_PRAGMA_UNROLL
    for (; key_start < p.num_keys;
         key_start += p.num_splits_key_device() * kBlockSizeJ) {
      output_frags.clear();

      int32_t next_key = key_start;
      int32_t query_start = getQueryStart(p, key_start);
      while (next_key == key_start && query_start < p.num_queries) {
        // This line here
        // vvvvvvvvvvvvvv
        warp_id = warp_uniform(warp_id);
        // ^^^^^^^^^^^^^^
        // ... makes everything use less RF and be 10% faster. Why?
        // I don't know. My theory is that it forces `nvcc` to
        // re-compute indices, offsets etc... and not keep them
        // from the previous iteration, which prevents MASSIVE
        // register spilling.

        processBlockIJ<kKeysQueriesAlignedToBlockSize>(
            shared_storage,
            output_frags,
            p,
            query_start,
            key_start,
            warp_id,
            lane_id);

        int32_t next_query;
        incrIteration(p, query_start, key_start, next_query, next_key);
        query_start = next_query;
      }
      if (kOutputInRF) {
        writeFragsToGmem<kKeysQueriesAlignedToBlockSize>(
            shared_storage, output_frags, p, key_start, warp_id, lane_id);
      } else if (getQueryStart(p, key_start) >= p.num_queries) {
        zfillGradKV<kKeysQueriesAlignedToBlockSize>(
            p, key_start, warp_id, lane_id);
      }
      __syncthreads();
    }
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void zfillGradKV(
      Params const& p,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    constexpr int kThreadsPerKey = 8;
    constexpr int kParallelKeys = kNumThreads / kThreadsPerKey;
    static_assert(kBlockSizeJ % kParallelKeys == 0, "");
    // This function is not really optimized, but should rarely be used
    // It's only used when some keys are "useless" and don't attend to
    // any query, due to causal masking

    int thread_id = 32 * warp_id + lane_id;
    int k_shift = lane_id % kThreadsPerKey;

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kBlockSizeJ; j += kParallelKeys) {
      int key = key_start + j + (thread_id / kThreadsPerKey);
      if (!skipBoundsChecks && key >= p.num_keys) {
        continue;
      }
      auto gv_ptr = p.grad_value_ptr + key * p.gV_strideM();
      auto gk_ptr = p.grad_key_ptr + key * p.gK_strideM();

      for (int k = k_shift; k < p.head_dim_value; k += kThreadsPerKey) {
        gv_ptr[k] = scalar_t(0);
      }
      for (int k = k_shift; k < p.head_dim; k += kThreadsPerKey) {
        gk_ptr[k] = scalar_t(0);
      }
    }
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void processBlockIJ(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params& p,
      int32_t query_start,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = p.scale;
    int16_t thread_id = 32 * warp_id + lane_id;

    auto rematerializeThreadIds = [&]() {
      // Prevents `nvcc` from keeping values deduced from
      // `thread_id`, `warp_id`, ... in RF - to reduce register pressure
      warp_id = warp_uniform(thread_id / 32);
      lane_id = thread_id % 32;
      thread_id = 32 * warp_id + lane_id;
    };

    bool isFirstQuery = (query_start == getQueryStart(p, key_start));
    int32_t next_query, next_key;
    incrIteration(p, query_start, key_start, next_query, next_key);
    bool isLastQuery = next_key != key_start;
    // PRINT_T0("[b:(%d, %d, %d)] q=%d k=%d FIRST=%d LAST=%d next: (%d, %d)",
    //   int(blockIdx.x), int(blockIdx.y), int(blockIdx.z),
    //   query_start, key_start, int(isFirstQuery), int(isLastQuery),
    //   int(next_query), int(next_key)
    //   );

    // NOTE(alih): we fetch lse_i here as well as d_i.
    accum_t lsei_rf = accum_t(0);
    accum_t di_rf = accum_t(0);
    if (thread_id < kBlockSizeI) {
      if (query_start + thread_id < p.num_queries) {
        auto query_offset = (query_start + thread_id) * p.num_heads;
        di_rf = p.delta_ptr[query_offset];
        lsei_rf = p.logsumexp_ptr[query_offset];
      }
      shared_storage.lse_i()[thread_id] = lsei_rf;
      shared_storage.di()[thread_id] = di_rf;
    }

    int32_t num_queries_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_queries - query_start));
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start));

    auto prologueGradV = [&](int col) {
      typename MatmulGradV::Mma::IteratorB iterator_dO(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);
      MatmulGradV::Mma::prologue(
          shared_storage.mm_gradV(),
          iterator_dO,
          thread_id,
          num_queries_in_block);
    };
    auto prologueGradQ = [&](int col) {
      typename MatmulGradQ::Mma::IteratorB iterator_K(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM + col,
          {num_keys_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradQ::Mma::prologue(
          shared_storage.mm_gradQ(), iterator_K, thread_id, num_keys_in_block);
    };
    auto prologueGradK = [&](int col) {
      typename MatmulGradK::Mma::IteratorB iterator_Q(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {num_queries_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradK::Mma::prologue(
          shared_storage.mm_gradK(),
          iterator_Q,
          thread_id,
          num_queries_in_block);
    };
    auto prologueDOV = [&]() {
      typename MatmulDOIVJ::Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);
      typename MatmulDOIVJ::Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},
          p.value_ptr + key_start * p.v_strideM,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);
      MatmulDOIVJ::Mma::prologue(
          shared_storage.mm_doivj(),
          iterator_A,
          iterator_B,
          thread_id,
          p.head_dim_value);
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulQK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          num_queries_in_block,
          p.head_dim // k
      );

      // k_j
      typename Mma::IteratorA iterator_A(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // q_i.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_qk_k(),
          shared_storage.mm_qk_q(),
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma.set_prologue_done(kPrologueQK);
      mma.set_zero_outside_bounds(!skipBoundsChecks);
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      accum = cutlass::multiplies<typename Mma::FragmentC>()(scale, accum);

      // Epilogue: add LSE + exp and store that to our shared memory buffer
      // shmem <- (matmul_result -
      // logsumexp[i_start:i_end].unsqueeze(1)).exp()
      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};

      auto lane_offset = MatmulQK::AccumLambdaIterator::get_lane_offset(
          lane_id, warp_id, output_tile_coords);

      MatmulQK::AccumLambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {},
          [&](int accum_m, int accum_n, int idx) {
            // NOTE(alih): Checking whether the current q coordinate is out of
            // bounds is something I had to do for SM70 and SM50 kernels in FNA
            // as well. Unsure why / and why xFormers doesn't (didn't) run into
            // issues with this.
            if (accum_m + key_start >= p.num_keys
#if __CUDA_ARCH__ < 800
                || accum_n + query_start >= p.num_queries
#endif
            ) {
              accum[idx] =
                  -cutlass::platform::numeric_limits<accum_t>::infinity();
            } else {
              accum[idx] -= shared_storage.lse_i()[accum_n];
            }
          },
          [&](int accum_m) {});

      __syncthreads();
      if (kPrologueGV) {
        prologueGradV(0);
      }
      if (kPrologueDOV) {
        prologueDOV();
      }

      // MatmulQK::B2bGemm::accumApplyLSEToSmem(
      MatmulQK::B2bGemm::accumApplyExpToSmem(
          shared_storage.attn_shared_storage(),
          accum,
          // p.logsumexp_ptr + query_start * p.num_heads,
          // problem_size.n(),
          thread_id,
          warp_id,
          lane_id,
          output_tile_coords);

      __syncthreads();
    }
    rematerializeThreadIds();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradV matmul
    //
    // grad_v[j_start:j_end] += attn_T @ do_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    constexpr bool kSingleIterationGradV =
        kMaxK <= MatmulGradV::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradV ? 1 : p.head_dim_value);
         col += MatmulGradV::ThreadblockShape::kN) {
      using Mma = typename MatmulGradV::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block, p.head_dim_value - col, num_queries_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradV::OutputTileIterator(
            typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
            p.grad_value_ptr + key_start * p.gV_strideM() + col,
            {num_keys_in_block, p.head_dim_value - col},
            thread_id);
      };
      typename Mma::IteratorB iterator_B(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);

      // dVj += Pij.T @ dOi
      Mma mma(
          // operand A: Pij.T
          shared_storage.attn_shared_storage().accum_ref(),
          //// operand A_scale Zij.T:
          //// if we're using dropout, operand A is Pij_dropped.T = Pij.T *
          /// Zij.T / which is computed on the fly as fragments of Pij.T are
          /// loaded in
          // shared_storage.zij().accum_ref(),
          //  operand B: dOi - which was loaded into shared memory previously
          //  when we computed dVj
          shared_storage.mm_gradV().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      int storage_id = col / MatmulGradV::ThreadblockShape::kN;
      AccumTileGmem gmem_tile{
          p.workspace_gv + storage_id * AccumTileGmem::kElementsStored};
      if (!kOutputInRF) {
        if (isFirstQuery || !kNeedsAccumGradV) {
          output_frags.gradV.clear();
        } else {
          gmem_tile.load(output_frags.gradV, thread_id);
        }
      }
      mma.set_prologue_done(kPrologueGV);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations,
          output_frags.gradV,
          iterator_B,
          output_frags.gradV);
      __syncthreads();
      if (kPrologueGV && !kSingleIterationGradV &&
          col + MatmulGradV::ThreadblockShape::kN < p.head_dim_value) {
        prologueGradV(col + MatmulGradV::ThreadblockShape::kN);
      }

      if (!kOutputInRF) {
        if (kNeedsAccumGradV && !isLastQuery) {
          gmem_tile.store(output_frags.gradV, thread_id);
        } else {
          accumulateInGmem<MatmulGradV>(
              shared_storage.gradV_epilogue(),
              output_frags.gradV,
              createEpilogueIter(),
              isFirstQuery || kNeedsAccumGradV,
              warp_id,
              lane_id);
        }
      }
    }
    __syncthreads();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulDOIVJ::Mma;
      // do_i
      typename Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},
          p.value_ptr + key_start * p.v_strideM,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);

      Mma mma(shared_storage.mm_doivj(), thread_id, warp_id, lane_id);
      mma.set_prologue_done(kPrologueDOV);
      mma.set_zero_outside_bounds(!skipBoundsChecks);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (p.head_dim_value + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      __syncthreads();
      if (kPrologueGQ) {
        prologueGradQ(0);
      }
      if (kPrologueGK) {
        prologueGradK(0);
      }

      int warp_idx_mn_0 =
          warp_id % (Mma::Base::WarpCount::kM * Mma::Base::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::Base::WarpCount::kM,
          warp_idx_mn_0 / Mma::Base::WarpCount::kM};
      // TODO: This must be terribly inefficient. There must be a better way
      // tmp [RF] <- (accum [RF] - Di [smem] ) * attn_T.T [smem]
      // attn_shared_storage  [smem] <- tmp.T
      // tmp_shared_storage [smem] <- tmp
      {
        using LambdaIterator = typename MatmulDOIVJ::AccumLambdaIterator;
        auto lane_offset = LambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);

        auto attn_T = shared_storage.attn_shared_storage().accum_ref();

        accum_t current_di;
        // dSij = (dPij - Di) * Pij
        LambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) { current_di = shared_storage.di()[accum_m]; },
            [&](int accum_m, int accum_n, int idx) {
              // TODO: Otherwise we can get nans as we
              // might have infs here (only seen on f16 tho)
              if (skipBoundsChecks ||
                  (accum_m < num_queries_in_block &&
                   accum_n < num_keys_in_block)) {
                accum_t attn = attn_T.at({accum_n, accum_m});
                accum[idx] = (accum[idx] - current_di) * attn;
              } else {
                accum[idx] = 0;
              }
            },
            [&](int accum_m) {

            });

        accum = accum * scale;

#if 0
        PRINT_B0_T0("(doivj - di) * attn * scale");
        print_warp_accum<LambdaIterator>(accum, lane_offset, 4, 4);
#endif

        __syncthreads();
        if (!MatmulGradK::DefaultMmaFromSmem::kIsTransposedA) {
          auto tmpT = shared_storage.tmpT_shared_storage().accum_ref();
          // attn <- attn_T.T
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m, int accum_n, int idx) {
                tmpT.at({accum_n, accum_m}) = scalar_t(accum[idx]);
              },
              [&](int accum_m) {});
        }
      }

      MatmulDOIVJ::B2bGemm::accumToSmem(
          shared_storage.tmp_shared_storage(),
          accum,
          lane_id,
          output_tile_coords);
      __syncthreads();
    }
    // Force `nvcc` to recompute values that depend on the variables just below
    // to use less RF and prevent some spilling
    p.head_dim = warp_uniform(p.head_dim);
    p.k_strideM = warp_uniform(p.k_strideM);
    rematerializeThreadIds();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradQ matmul
    //
    // grad_q[i_start:i_end] += tmp @ k_j
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Skip the loop & associated branches if we know at compile time the number
    // of iterations
    constexpr bool kSingleIterationGradQ =
        kMaxK <= MatmulGradQ::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradQ ? 1 : p.head_dim);
         col += MatmulGradQ::ThreadblockShape::kN) {
      using Mma = typename MatmulGradQ::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      cutlass::gemm::GemmCoord problem_size(
          num_queries_in_block,
          false ? MatmulGradQ::ThreadblockShape::kN : p.head_dim - col,
          num_keys_in_block);

      // k_j
      typename Mma::IteratorB iterator_B(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      auto a = shared_storage.tmp_shared_storage().accum_ref();
      Mma mma(
          // operand A: dSij
          shared_storage.tmp_shared_storage().accum_ref(),
          // operand B: Kj
          shared_storage.mm_gradQ().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      int col_id = col / MatmulGradQ::ThreadblockShape::kN;
      int num_cols = kSingleIterationGradQ
          ? 1
          : ceil_div(p.head_dim, MatmulGradQ::ThreadblockShape::kN);
      int storage_id = (col_id + query_start / kBlockSizeI * num_cols);

      if (p.num_splits_key_device() > 1) {
        AtomicLock::acquire(
            &p.workspace_gq[storage_id].lock,
            p.split_key_device() + 1,
            thread_id);
        // Make sure we can see other block's output
        __threadfence();
      }

      AccumTileGmem gmem_tile{&p.workspace_gq[storage_id].buffer[0]};
      if (!kNeedsAccumGradQ ||
          (p.num_splits_key_device() == 1 && key_start == 0)) {
        // if we know we are the first to access it, we know it's only zeros.
        // Avoids a load from gmem (and gmem init as well)
        accum.clear();
      } else {
        gmem_tile.load(accum, thread_id);
      }

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();
      mma.set_prologue_done(kPrologueGQ);
      mma(gemm_k_iterations, accum, iterator_B, accum);
      __syncthreads();
      bool isLastColumn = kSingleIterationGradQ ||
          (col + MatmulGradQ::ThreadblockShape::kN >= p.head_dim);
      if (kPrologueGQ && !isLastColumn) {
        prologueGradQ(col + MatmulGradQ::ThreadblockShape::kN);
      }

      bool isLast = [&]() {
        int32_t next_key = key_start + p.num_splits_key_device() * kBlockSizeJ;
        if (p.num_keys <= next_key) {
          return true;
        }
        if (query_start < getSmallestQueryForKey(p, next_key)) {
          return true;
        }
        return false;
      }();
      // Output results
      if (p.num_splits_key_device() > 1) {
        int32_t numAddsSoFar = -1;
        if (isLast && thread_id == 0) {
          numAddsSoFar = atomicAdd(&p.workspace_gq[storage_id].counter, 1) +
              1; // `atomicAdd` returns the old value
        }
        isLast = __syncthreads_or(
            numAddsSoFar == getNumParallelBlocksForQuery(p, query_start));
        assert(numAddsSoFar <= getNumParallelBlocksForQuery(p, query_start));
      }
      if (kNeedsAccumGradQ && !isLast) {
        gmem_tile.store(accum, thread_id);
        if (p.num_splits_key_device() > 1) {
          // Make sure everyone wrote before we release the lock
          __threadfence();
          __syncthreads();
          AtomicLock::release(&p.workspace_gq[storage_id].lock, thread_id);
        }
      } else {
        // NOTE: We're not releasing the lock because no one is expected
        // to come after us (we're the last one to write)
        typename MatmulGradQ::OutputTileIterator output_it(
            typename MatmulGradQ::OutputTileIterator::Params{p.gQ_strideM()},
            p.grad_query_ptr + query_start * p.gQ_strideM() + col,
            {problem_size.m(), problem_size.n()},
            thread_id);
        // if `direct_store` is True, we store to gmem (`*gmem = accum`)
        // otherwise, we accumulate in gmem (`*gmem = *gmem + accum`)
        // If we know ahead of time when we will write for the first time
        // we can:
        // (1) Avoid an additional memory read
        // (2) Avoid the cost of initializing memory to 0
        bool direct_store = kNeedsAccumGradQ || key_start == 0 ||
            (p.num_splits_key_device() > 1);
        accumulateInGmem<MatmulGradQ>(
            isLastColumn ? shared_storage.gradQ_epilogue_lastIter()
                         : shared_storage.gradQ_epilogue(),
            accum,
            output_it,
            direct_store,
            warp_id,
            lane_id);
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradK matmul
    //
    // grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    rematerializeThreadIds();

    constexpr bool kSingleIterationGradK =
        kMaxK <= MatmulGradK::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradK ? 1 : p.head_dim);
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col,
          num_queries_in_block);
      auto createEpilogueIter = [&]() {
        return typename MatmulGradK::OutputTileIterator(
            typename MatmulGradK::OutputTileIterator::Params{p.gK_strideM()},
            p.grad_key_ptr + key_start * p.gK_strideM() + col,
            {num_keys_in_block,
             false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col},
            thread_id);
      };

      // q_i
      typename Mma::IteratorB iterator_B(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      auto getTmp = [&](int) { return &shared_storage.tmp_shared_storage(); };
      auto getTmpT = [&](int) { return &shared_storage.tmpT_shared_storage(); };
      // this is basically:
      // opA = kIsTransposedA ? getTmp() : getTmpT();
      bool constexpr kIsTransposedA =
          MatmulGradK::DefaultMmaFromSmem::kIsTransposedA;
      auto& opA = *call_conditional<
          kIsTransposedA,
          decltype(getTmp),
          decltype(getTmpT)>::apply(getTmp, getTmpT, 0);
      Mma mma(
          // operand A: dSij.T
          opA.accum_ref(),
          // operand B: Qi
          shared_storage.mm_gradK().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      int storage_id = col / MatmulGradK::ThreadblockShape::kN;
      AccumTileGmem gmem_tile{
          p.workspace + storage_id * AccumTileGmem::kElementsStored};
      if (!kOutputInRF) {
        if (isFirstQuery || !kNeedsAccumGradK) {
          output_frags.gradK.clear();
        } else {
          gmem_tile.load(output_frags.gradK, thread_id);
        }
      }
      mma.set_prologue_done(kPrologueGK);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      __syncthreads();

      mma(gemm_k_iterations,
          output_frags.gradK,
          iterator_B,
          output_frags.gradK);
      __syncthreads();
      bool isLastColumn = kSingleIterationGradK ||
          col + MatmulGradK::ThreadblockShape::kN >= p.head_dim;
      if (kPrologueGK && !isLastColumn) {
        prologueGradK(col + MatmulGradK::ThreadblockShape::kN);
      }

      if (kPrologueQK && isLastColumn) {
        int32_t next_query, next_key;
        incrIteration(p, query_start, key_start, next_query, next_key);
        DISPATCH_BOOL(
            next_key != key_start, kForceReloadK, ([&]() {
              prologueQkNextIteration<kForceReloadK>(
                  shared_storage, p, next_query, next_key, warp_id, lane_id);
            }));
      }

      // Output results
      if (!kOutputInRF) {
        if (kNeedsAccumGradK && !isLastQuery) {
          gmem_tile.store(output_frags.gradK, thread_id);
        } else {
          accumulateInGmem<MatmulGradK>(
              isLastColumn ? shared_storage.gradK_epilogue_final()
                           : shared_storage.gradK_epilogue(),
              output_frags.gradK,
              createEpilogueIter(),
              isFirstQuery || kNeedsAccumGradK,
              warp_id,
              lane_id);
          __syncthreads();
        }
      }
    }
  }

  static CUTLASS_HOST_DEVICE int32_t getQueryStartShift(Params const& p) {
    if (p.num_splits_key_device() > 1) {
      return (p.split_key_device() * kBlockSizeI) % getQueryEnd(p);
    }
    return 0;
  }

  // Iteration order logic
  static CUTLASS_HOST_DEVICE int32_t
  getQueryStart(Params const& p, int32_t key_start) {
    return getSmallestQueryForKey(p, key_start) + getQueryStartShift(p);
  };
  static CUTLASS_HOST_DEVICE int32_t getQueryEnd(Params const& p) {
    return align_up(p.num_queries, kBlockSizeI);
  };

  static CUTLASS_HOST_DEVICE int32_t
  getSmallestQueryForKey(Params const& p, int32_t key_start) {
    return 0;
  };

  // Returns how many kernel blocks will write to a given block in `grad_query`
  // This is usually equal to the number of key splits, but can be different
  // for instance in the causal case, or varying seqlen
  static CUTLASS_HOST_DEVICE int32_t
  getNumParallelBlocksForQuery(Params const& p, int32_t query_start) {
    int16_t num_key_blocks = ceil_div(p.num_keys, kBlockSizeJ);
    return cutlass::fast_min(p.num_splits_key_device(), num_key_blocks);
  };

  // Returns the next block to process
  static CUTLASS_HOST_DEVICE void incrIteration(
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      int32_t& next_query,
      int32_t& next_key) {
    next_query = query_start + kBlockSizeI;
    next_key = key_start;
    auto query_shift = getQueryStartShift(p);
    // Wrap around
    if (query_shift) {
      if (next_query >= p.num_queries) {
        next_query = getSmallestQueryForKey(p, key_start);
        return;
      } else if (query_start < query_shift && query_shift <= next_query) {
        // jump to next key
      } else {
        return;
      }
    } else {
      if (next_query < p.num_queries) {
        return;
      }
      // jump to next key
    }
    // Next key
    next_key = key_start + p.num_splits_key_device() * kBlockSizeJ;
    next_query = getQueryStart(p, next_key);
  }

  template <bool kForceReloadK>
  static CUTLASS_DEVICE void prologueQkNextIteration(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    if (query_start >= p.num_queries || key_start >= p.num_keys) {
      return;
    }

    static constexpr bool kReloadK =
        kForceReloadK || !MatmulQK::Mma::kSmemContainsEntireMat;
    int thread_id = 32 * warp_id + lane_id;
    typename MatmulQK::Mma::IteratorA iterator_A(
        {int32_t(p.k_strideM)},
        p.key_ptr + key_start * p.k_strideM,
        {p.num_keys - key_start, p.head_dim},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    typename MatmulQK::Mma::IteratorB iterator_B(
        {int32_t(p.q_strideM)},
        p.query_ptr + query_start * p.q_strideM,
        {p.head_dim, p.num_queries - query_start},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    MatmulQK::Mma::prologue<kReloadK, true>(
        shared_storage.mm_qk_k(),
        shared_storage.mm_qk_q(),
        iterator_A,
        iterator_B,
        thread_id,
        p.head_dim);
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void writeFragsToGmem(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params const& p,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    uint16_t thread_id = 32 * warp_id + lane_id;
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_keys - key_start);
    typename MatmulGradV::OutputTileIterator outputV_it(
        typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
        p.grad_value_ptr + key_start * p.gV_strideM(),
        {num_keys_in_block, p.head_dim_value},
        thread_id);
    accumulateInGmem<MatmulGradV>(
        shared_storage.gradV_epilogue_final(),
        output_frags.gradV,
        outputV_it,
        true,
        warp_id,
        lane_id);

    typename MatmulGradK::OutputTileIterator outputK_it(
        typename MatmulGradK::OutputTileIterator::Params{p.gK_strideM()},
        p.grad_key_ptr + key_start * p.gK_strideM(),
        {num_keys_in_block,
         false ? MatmulGradK::ThreadblockShape::kN : p.head_dim},
        thread_id);
    accumulateInGmem<MatmulGradK>(
        shared_storage.gradK_epilogue_final(),
        output_frags.gradK,
        outputK_it,
        true,
        warp_id,
        lane_id);
  }

  template <typename MatmulT>
  static CUTLASS_DEVICE void accumulateInGmem(
      typename MatmulT::DefaultEpilogue::SharedStorage& epilogue_smem,
      typename MatmulT::Mma::FragmentC const& accum,
      typename MatmulT::OutputTileIterator output_it,
      bool first,
      uint8_t warp_id,
      uint8_t lane_id) {
    using DefaultEpilogue = typename MatmulT::DefaultEpilogue;
    using DefaultOutputOp = typename MatmulT::DefaultOutputOp;
    using Mma = typename MatmulT::Mma;
    int thread_id = 32 * warp_id + lane_id;
    DISPATCH_BOOL(
        first, kIsFirst, ([&]() {
          static constexpr auto ScaleType = kIsFirst
              ? cutlass::epilogue::thread::ScaleType::Nothing
              : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
          using EpilogueOutputOp =
              typename cutlass::epilogue::thread::LinearCombination<
                  typename DefaultOutputOp::ElementOutput,
                  DefaultOutputOp::kCount,
                  typename DefaultOutputOp::ElementAccumulator,
                  typename DefaultOutputOp::ElementCompute,
                  ScaleType>;
          using Epilogue =
              typename cutlass::epilogue::threadblock::EpiloguePipelined<
                  typename DefaultEpilogue::Shape,
                  typename Mma::Operator,
                  DefaultEpilogue::kPartitionsK,
                  typename MatmulT::OutputTileIterator,
                  typename DefaultEpilogue::AccumulatorFragmentIterator,
                  typename DefaultEpilogue::WarpTileIterator,
                  typename DefaultEpilogue::SharedLoadIterator,
                  EpilogueOutputOp,
                  typename DefaultEpilogue::Padding,
                  DefaultEpilogue::kFragmentsPerIteration,
                  true // IterationsUnroll
                  >;
          EpilogueOutputOp rescale({1, 1});
          Epilogue epilogue(epilogue_smem, thread_id, warp_id, lane_id);
          epilogue(rescale, output_it, accum, output_it);
        }));
  }

  template <int kElementsPerAccess>
  static CUTLASS_DEVICE void computeDelta(
      Params const& p,
      int32_t query_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    // Each thread computes one value for Delta
    // Depending on warp configuration, we might have multiple
    // threads of the same warp working on the same row
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static_assert(kNumThreads >= kBlockSizeI, "");
    static constexpr int kNumThreadsPerLine = kNumThreads / kBlockSizeI;
    int16_t thread_id = 32 * warp_id + lane_id;

    int16_t laneFirstCol = kElementsPerAccess * (lane_id % kNumThreadsPerLine);
    int16_t laneRow = thread_id / kNumThreadsPerLine;
    bool rowPred = (query_start + laneRow) < p.num_queries;
    bool pred = rowPred;

    // on windows, previous syntax __restrict__ AccessType*
    // resulted in error: "restrict" is not allowed
    const AccessType* __restrict__ grad_output_ptr =
        reinterpret_cast<const AccessType*>(
            p.grad_output_ptr + (query_start + laneRow) * p.gO_strideM +
            laneFirstCol);
    const AccessType* __restrict__ output_ptr =
        reinterpret_cast<const AccessType*>(
            p.output_ptr + (query_start + laneRow) * p.o_strideM() +
            laneFirstCol);

    static constexpr int64_t kMaxIters =
        kMaxK / (kElementsPerAccess * kNumThreadsPerLine);
    constexpr int kPipelineStages = 2;
    accum_t delta_value = accum_t(0);
    using GlobalLoad =
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>;
    AccessType frag_grad_output[kPipelineStages];
    AccessType frag_output[kPipelineStages];

    auto loadAndIncrement = [&](int ld_pos, bool is_valid) {
      frag_grad_output[ld_pos].clear();
      frag_output[ld_pos].clear();
      GlobalLoad(frag_grad_output[ld_pos], grad_output_ptr, is_valid);
      GlobalLoad(frag_output[ld_pos], output_ptr, is_valid);
      grad_output_ptr += kNumThreadsPerLine;
      output_ptr += kNumThreadsPerLine;
    };

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kPipelineStages - 1; ++iter) {
      int ld_pos = iter % kPipelineStages;
      pred = pred &&
          (laneFirstCol + iter * kElementsPerAccess * kNumThreadsPerLine) <
              p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
    }
    auto columnIteration = [&](int iter) {
      // Load for next iter
      int ld_pos = (iter + kPipelineStages - 1) % kPipelineStages;
      pred = pred &&
          (laneFirstCol +
           (iter + kPipelineStages - 1) * kElementsPerAccess *
               kNumThreadsPerLine) < p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccessType::kElements; ++i) {
        delta_value += accum_t(frag_output[iter % kPipelineStages][i]) *
            accum_t(frag_grad_output[iter % kPipelineStages][i]);
      }
    };

    // If we have a small lower-bound for K, we can unroll the loop
    if (kMaxK <= 256) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kMaxIters; ++iter) {
        columnIteration(iter);
      }
    } else {
      int num_iters =
          ceil_div(p.head_dim_value, kElementsPerAccess * kNumThreadsPerLine) *
          (kElementsPerAccess * kNumThreadsPerLine);
      for (int iter = 0; iter < num_iters; ++iter) {
        columnIteration(iter);
      }
    }

    // Reduce between workers
    static_assert(
        kNumThreadsPerLine == 1 || kNumThreadsPerLine == 2 ||
            kNumThreadsPerLine == 4,
        "");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
      delta_value = delta_value + __shfl_xor_sync(0xffffffff, delta_value, i);
    }

    // Store in gmem
    if (rowPred) {
      p.delta_ptr[(query_start + laneRow) * p.num_heads] = delta_value;
    }
  }
};

} // namespace fmha
} // namespace cuda
} // namespace natten
