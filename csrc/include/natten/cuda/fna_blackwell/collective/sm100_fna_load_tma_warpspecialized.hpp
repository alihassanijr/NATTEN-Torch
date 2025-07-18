/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "natten/cuda/fna_blackwell/collective/fna_common.hpp"
#include "natten/cuda/fna_blackwell/collective/fna_fusion.hpp"

namespace cutlass::fna::collective {

using namespace cute;

template <
    class Element,
    class StrideQ,
    class StrideK,
    class StrideV,
    class CollectiveMmaQK,
    class CollectiveMmaPV,
    class SmemLayoutQ,
    class SmemLayoutK,
    class SmemLayoutV,
    class TensorStorage,
    class PipelineQ,
    class PipelineKV,
    class Mask,
    class TileShape,
    class QTileShape,
    class KVTileShape,
    class NADim>
struct Sm100FnaLoadTmaWarpspecialized {
  using MultiDimTileShape = cute::tuple<QTileShape, KVTileShape>;

  static_assert(
      size(QTileShape{}) == get<0>(TileShape{}),
      "QTileShape doesn't match the size of Q tile in the FMHA kernel.");
  static_assert(
      size(KVTileShape{}) == get<1>(TileShape{}),
      "QTileShape doesn't match the size of Q tile in the FMHA kernel.");

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  struct Arguments {
    const Element* ptr_Q;
    StrideQ dQ;
    const Element* ptr_K;
    StrideK dK;
    const Element* ptr_V;
    StrideV dV;
    NADim q_shape;
    NADim kv_shape;
    NADim qkv_shape;
    NADim window_size;
    NADim stride;
    NADim dilation;
    int num_heads_actual; // heads / size(dilation)
    int num_extra_kv;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
    NADim qkv_shape;
    NADim q_shape;
    NADim kv_shape;
    cute::tuple<NADim, NADim, NADim, NADim>
        na_params; // win, win_left, win_right, stride
    int num_extra_kv;
    int extra_kv_offset;
    bool is_fully_block_sparse;
    bool has_kv_padding;
    bool extra_kv_divides_tile;
    NADim dilation;
    bool requires_qkv_fixup;
    bool is_dilated;
    int num_heads_actual;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {
    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    auto dQ = args.dQ;
    auto dK = args.dK;
    auto dV = args.dV;
    auto problem_shape_qk = problem_shape;

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        problem_shape_qk,
        typename CollectiveMmaQK::Arguments{
            ptr_Q,
            dQ,
            ptr_K,
            dK,
        },
        /*workspace=*/nullptr);

    auto problem_shape_pv = select<0, 2, 1, 3>(problem_shape_qk);
    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments{
            ptr_K,
            dK, // never used, dummy
            ptr_V,
            select<1, 0, 2>(dV),
        },
        /*workspace=*/nullptr);

    auto window_left = get_window_left(args.window_size);
    auto window_right = get_window_right(args.window_size);

    bool requires_qkv_fixup = not evenly_divides(args.qkv_shape, args.dilation);

    return Params{
        params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b,
        args.qkv_shape,
        args.q_shape,
        args.kv_shape,
        make_tuple(args.window_size, window_left, window_right, args.stride),
        args.num_extra_kv,
        /* extra_kv_offset */ size(args.kv_shape), // NOTE: dilation > 1 would
                                                   // have to repeat extra kv
                                                   // along heads
        fully_block_sparse<typename Mask::Causal>(
            args.qkv_shape,
            args.window_size,
            args.stride,
            QTileShape{},
            KVTileShape{}),
        /* has_kv_padding */ not evenly_divides(args.qkv_shape, KVTileShape{}),
        /* extra_kv_divides_tile */ args.num_extra_kv % get<1>(TileShape{}) ==
            0,
        args.dilation,
        requires_qkv_fixup,
        is_dilated(args.dilation),
        args.num_heads_actual};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void load(
      BlkCoord const& blk_coord_in,
      ProblemShape const& problem_shape,
      Params const& params,
      ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q,
      typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv,
      typename PipelineKV::PipelineState& pipeline_kv_producer_state) {
    BlkCoord blk_coord_q = blk_coord_in;
    BlkCoord blk_coord_kv = blk_coord_in;

    auto qkv_shape = params.qkv_shape;
    if (params.requires_qkv_fixup) {
      qkv_shape = Mask{}.correct_qkv_shape(
          problem_shape,
          params.qkv_shape,
          blk_coord_in,
          params.dilation,
          params.num_heads_actual);
    } else if (params.is_dilated) {
      qkv_shape = ceil_div(params.qkv_shape, params.dilation);
    }

    auto [kv_start, num_tiles] = Mask{}.get_trip_count(
        blk_coord_in,
        MultiDimTileShape{},
        params.q_shape,
        qkv_shape,
        params.na_params);

    int mask_tile_count = size(num_tiles);

    auto mask_tile_count_extra =
        Mask{}.get_extra_trip_info(MultiDimTileShape{}, params.num_extra_kv);

    int extra_kv_tile_offset =
        ceil_div(params.extra_kv_offset, get<1>(TileShape{}));

    auto kv_start_tile = ceil_div(kv_start, KVTileShape{});

    auto kv_tiled = ceil_div(params.kv_shape, KVTileShape{});
    auto ctr = make_identity_tensor(num_tiles);
    auto ctr_offset = domain_offset(kv_start_tile, ctr);

    auto kv_tiled_layout = make_layout(kv_tiled);

    using X = Underscore;

    // this one is only executed by one thread, no need to elect_one

    // Q1, K1, Q2, V1, K2, V2, K3, V3, ...
    // two pipes: Q and KV
    // from Memory (prod) to TensorCore (cons)

    // compute gQ, sQ
    // we load 2*get<0>(blk_coord), and 2*get<0>(blk_coord) + 1
    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    Tensor mQ_qdl_p =
        params.tma_load_q.get_tma_tensor(select<0, 2, 3>(problem_shape));

    int q_offs_0 = 0;
    int q_offs_2_1 = 0;

    Tensor mQ_qdl = domain_offset(
        make_coord(q_offs_0, _0{}, make_coord(_0{}, q_offs_2_1)), mQ_qdl_p);

    Tensor gQ_qdl = local_tile(
        mQ_qdl, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ_qdl);
    Tensor sQ =
        make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto [tQgQ_qdl, tQsQ] = tma_partition(
        params.tma_load_q,
        _0{},
        make_layout(_1{}),
        group_modes<0, 3>(sQ),
        group_modes<0, 3>(tSgQ_qdl));
    Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord_q));

    // compute gK, sK
    Tensor mK_kdl_p =
        params.tma_load_k.get_tma_tensor(select<1, 2, 3>(problem_shape));

    int kv_offs_0 = 0;
    int kv_offs_2_1 = 0;

    Tensor mK_kdl = domain_offset(
        make_coord(kv_offs_0, _0{}, make_coord(_0{}, kv_offs_2_1)), mK_kdl_p);

    Tensor gK_kdl = local_tile(
        mK_kdl, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tSgK_kdl = mma_qk.partition_B(gK_kdl);
    Tensor sK =
        make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    auto [tKgK_kdl, tKsK] = tma_partition(
        params.tma_load_k,
        _0{},
        make_layout(_1{}),
        group_modes<0, 3>(sK),
        group_modes<0, 3>(tSgK_kdl));
    Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord_kv));

    // compute gV, sV
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor mV_dkl_p =
        params.tma_load_v.get_tma_tensor(select<2, 1, 3>(problem_shape));

    Tensor mV_dkl = domain_offset(
        make_coord(_0{}, kv_offs_0, make_coord(_0{}, kv_offs_2_1)), mV_dkl_p);

    Tensor gV_dkl = local_tile(
        mV_dkl, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tOgV_dkl = mma_pv.partition_B(gV_dkl);
    Tensor sV =
        make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    auto [tVgV_dkl, tVsV] = tma_partition(
        params.tma_load_v,
        _0{},
        make_layout(_1{}),
        group_modes<0, 3>(sV),
        group_modes<0, 3>(tOgV_dkl));
    auto tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord_kv));

    // blk_coord in decomposed in terms of TileShape, not TileShapeQK
    // As such, it needs to be transformed as
    // (a,b,c): a -> 2*a (Q0) 2*a+1 (Q1)
    //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)

    uint32_t lane_predicate = cute::elect_one_sync();

    // Q1
    int q0_index = 2 * get<0>(blk_coord_q);
    int q1_index = 2 * get<0>(blk_coord_q) + 1;
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier =
          pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(
          params.tma_load_q.with(*tma_barrier, 0),
          tQgQ(_, q0_index),
          tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // K1
    int k_index = 0;
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier =
          pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(
          params.tma_load_k.with(*tma_barrier, 0),
          tKgK(_, crd2idx(ctr_offset(k_index), kv_tiled_layout)),
          tKsK(_, pipeline_kv_producer_state.index()));
    }
    ++pipeline_kv_producer_state;

    // Q2
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier =
          pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(
          params.tma_load_q.with(*tma_barrier, 0),
          tQgQ(_, q1_index),
          tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // V1
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier =
          pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(
          params.tma_load_v.with(*tma_barrier, 0),
          tVgV(_, crd2idx(ctr_offset(k_index), kv_tiled_layout)),
          tVsV(_, pipeline_kv_producer_state.index()));
    }
    ++pipeline_kv_producer_state;
    k_index += 1;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      // Ki
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier =
            pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(
            params.tma_load_k.with(*tma_barrier, 0),
            tKgK(_, crd2idx(ctr_offset(k_index), kv_tiled_layout)),
            tKsK(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;

      // Vi
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier =
            pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(
            params.tma_load_v.with(*tma_barrier, 0),
            tVgV(_, crd2idx(ctr_offset(k_index), kv_tiled_layout)),
            tVsV(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;
      k_index += 1;
    }

    k_index = extra_kv_tile_offset;
    for (; mask_tile_count_extra > 0; mask_tile_count_extra -= 1) {
      // Ki
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier =
            pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(
            params.tma_load_k.with(*tma_barrier, 0),
            tKgK(_, k_index),
            tKsK(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;

      // Vi
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier =
            pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(
            params.tma_load_v.with(*tma_barrier, 0),
            tVgV(_, k_index),
            tVsV(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;
      k_index += 1;
    }
  }
};

} // namespace cutlass::fna::collective
