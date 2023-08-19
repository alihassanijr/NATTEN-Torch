#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks15_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks17_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks19_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks21_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks23_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks25_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks27_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks29_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks31_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks33_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks15_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks17_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks19_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks21_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks23_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks25_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks27_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks29_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks31_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks33_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks15_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks17_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks19_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks21_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks23_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks25_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks27_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks29_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks31_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks33_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33_align1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

