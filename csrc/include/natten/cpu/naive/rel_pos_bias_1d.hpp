/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/
/*! \file
    \brief Relative positional bias backward pass CPU kernel for 1D data.
*/

#pragma once
// TODO: these kernels should be independent of torch api.
// But for now, we do need vectorized reads.
#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

#include "natten/cpu/naive/natten_cpu_commons.h"

namespace natten {
namespace cpu {
namespace naive {

#define GRAIN_SIZE 0

// TODO: AVX

template <typename scalar_t>
struct RelPosBiasGradient1D {

  using idx_t = const int64_t;

  void operator()(
    void * d_bias_ptr,
    void * d_attn_ptr,
    int batch_size,
    int heads,
    int length,
    int dim,
    int kernel_size,
    int dilation,
    void * kv_seq_len) {
    launch(
      reinterpret_cast<scalar_t*>(d_bias_ptr),
      reinterpret_cast<scalar_t*>(d_attn_ptr),
      length, heads, kernel_size, dilation, batch_size, 
      reinterpret_cast<idx_t*>(kv_seq_len));
  }

  void launch(
    scalar_t* d_bias,
    scalar_t* d_attn,
    const int length,
    const int heads,
    const int kernel_size,
    const int dilation,
    const int batch_size,
    idx_t* kv_seq_len) {
    const int neighborhood_size = kernel_size / 2;
    const int d_bias_stride_0 = 2 * kernel_size - 1;
    const int d_attn_stride_2 = kernel_size;
    const int d_attn_stride_1 = length * d_attn_stride_2;
    const int d_attn_stride_0 = heads * d_attn_stride_1;
    for (int b = 0; b < batch_size; ++b) {
      const int unpadded_length = (kv_seq_len == nullptr) ? length : std::max(idx_t(kernel_size * dilation), kv_seq_len[b]);
      for (int i = 0; i < unpadded_length; i++) {
        const int pi = get_pb_start(i, unpadded_length, kernel_size, neighborhood_size, dilation);
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int ki = 0; ki < kernel_size; ki++) {
              d_bias[h * d_bias_stride_0 + (pi+ki)] += d_attn[b * d_attn_stride_0 + h * d_attn_stride_1 + i * d_attn_stride_2 + ki];
            }
        }});
      }
    }
  }
};

} // namespace naive
} // namespace cpu 
} // namespace natten
