# Copyright (c) 2022 - 2026 Ali Hassani.
#
# This script is intended to emit fused kernel instantiations into
# a variable number of source files, generate appropriate headers
# and a single dispatcher interface, which will be used by the
# NATTEN API to call the kernels.
#
# NOTE: these scripts are heavily under-documented, and
# overly-repetitive, and will be replaced in future PRs.
# Please use it with caution.

import argparse
import os
from typing import List

DEFAULT_OUTPUT_DIR = "csrc/"


KERNEL_DECL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_kv,
      int heads,
      int heads_kv,
      int dim,
      int dim_value,
      float attn_scale,
      cudaStream_t stream);
"""


KERNEL_IMPL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_kv,
      int heads,
      int heads_kv,
      int dim,
      int dim_value,
      float attn_scale,
      cudaStream_t stream) {{

  fmha_reference_forward<{Causal}>(
    static_cast<{dtype}*>(ptr_Q),
    static_cast<{dtype}*>(ptr_K),
    static_cast<{dtype}*>(ptr_V),
    static_cast<{dtype}*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen_q,
    seqlen_kv,
    heads,
    heads_kv,
    dim,
    dim_value,
    attn_scale,
    stream);
}}
"""


KERNEL_BWD_DECL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_kv,
      int heads,
      int heads_kv,
      int dim,
      int dim_value,
      float attn_scale,
      cudaStream_t stream);
"""


KERNEL_BWD_IMPL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_kv,
      int heads,
      int heads_kv,
      int dim,
      int dim_value,
      float attn_scale,
      cudaStream_t stream) {{

  fmha_reference_backward<{Causal}>(
    static_cast<{dtype}*>(ptr_Q),
    static_cast<{dtype}*>(ptr_K),
    static_cast<{dtype}*>(ptr_V),
    static_cast<{dtype}*>(ptr_O),
    static_cast<{dtype}*>(ptr_DO),
    static_cast<{dtype}*>(ptr_DQ),
    static_cast<{dtype}*>(ptr_DK),
    static_cast<{dtype}*>(ptr_DV),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen_q,
    seqlen_kv,
    heads,
    heads_kv,
    dim,
    dim_value,
    attn_scale,
    stream);
}}
"""


class DataType:
    def __init__(self, name, short_name, torch_name, bits):
        self.name = name
        self.bits = bits
        self.short_name = short_name
        self.torch_name = torch_name


Float = DataType("float", "float32", "torch::kFloat32", 32)
Half = DataType("cutlass::half_t", "float16", "torch::kFloat16", 16)
BFloat = DataType("cutlass::bfloat16_t", "bfloat16", "torch::kBFloat16", 16)


class ReferenceFmhaInstance:
    def __init__(
        self,
        dtype: DataType,
        causal: tuple,
        is_backward: bool,
    ):
        self.causal = causal
        self.dtype = dtype
        self.is_backward = is_backward

    def get_causal_cute(self) -> str:
        return "true" if self.causal else "false"

    def get_name(self) -> str:
        backward_str = "forward" if not self.is_backward else "backward"
        name = f"reference_fmha_{backward_str}"
        name += f"_{self.dtype.short_name}"
        name += "_causal" if self.causal else ""
        return name

    def get_decl(self) -> str:
        return (
            KERNEL_BWD_DECL_TEMPLATE if self.is_backward else KERNEL_DECL_TEMPLATE
        ).format(
            kernel_name=self.get_name(),
            dtype=self.dtype.name,
        )

    def get_impl(self) -> str:
        return (
            KERNEL_BWD_IMPL_TEMPLATE if self.is_backward else KERNEL_IMPL_TEMPLATE
        ).format(
            kernel_name=self.get_name(),
            Causal=self.get_causal_cute(),
            dtype=self.dtype.name,
        )


def write_combined_source_file(path, filename, headers, kernels):
    source_head = []
    source_head += ["#ifdef NATTEN_WITH_CUTLASS\n"]

    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]

    source_head += ["#include <ATen/ATen.h>\n"]
    source_head += ["#include <ATen/cuda/CUDAContext.h>\n"]
    source_head += ["#include <c10/cuda/CUDAGuard.h>\n"]
    source_head += ["#include <c10/cuda/CUDAStream.h>\n"]
    source_head += ["#include <torch/extension.h>\n"]

    source_head += ["#include <natten/natten.h>\n"]
    source_head += ["#include <natten/helpers.h>\n"]

    source_head += ["#include <natten/cuda/reference/fmha_reference_forward.hpp>\n"]
    source_head += ["#include <natten/cuda/reference/fmha_reference_backward.hpp>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace reference { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for kernel in kernels:
        source_body += "\n\n" + kernel.get_impl() + "\n\n"
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} // namespace reference \n",
            "} // namespace cuda \n",
            "} // namespace natten \n",
            "#endif \n",
            "\n",
        ]
    )
    filename = f"{path}/{filename}"
    with open(filename, "w") as f:
        f.write(source_head)
        f.write(source_body)
        f.write(source_foot)


class DTypeDispatcher:
    def __init__(self, is_backward: bool):
        self.dtypes: List[DataType] = []
        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_REFERENCE_FMHA_{fwd_bwd_str}"
        self.is_backward = is_backward

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_target_name(self, dtype, is_causal):
        kernel = self.get_kernel_instance(dtype, is_causal)
        return kernel.get_name()

    def get_kernel_instance(self, dtype, is_causal):
        return ReferenceFmhaInstance(
            dtype=dtype,
            causal=is_causal,
            is_backward=self.is_backward,
        )

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dtype, is_causal, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += "  if (is_causal) { "
            dispatcher_str += f"natten::cuda::reference::{self.get_target_name(dtype, True)}(__VA_ARGS__); \\\n"
            dispatcher_str += "  } else { "
            dispatcher_str += f"natten::cuda::reference::{self.get_target_name(dtype, False)}(__VA_ARGS__); \\\n"
            dispatcher_str += "  }"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Reference FMHA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + "Reference FMHA does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


def write_header_file(content, path, namespaces, extra_includes=None):
    extra_includes = extra_includes or []
    header_head = [
        "#pragma once\n",
        "\n\n",
    ]
    header_head += ["#include <iostream> \n"]
    header_head += ["#include <type_traits> \n"]
    header_head += ["#ifdef NATTEN_WITH_CUTLASS\n"]
    for incl in extra_includes:
        header_head += [f"#include <{incl}> \n"]

    for namespace in namespaces:
        header_head += [f"namespace {namespace}", " { \n"]

    header_foot = [
        "\n\n",
    ]
    for namespace in namespaces:
        header_foot += ["} ", f"// namespace {namespace}", " \n"]
    header_foot += [
        "#endif \n",
        "\n",
    ]
    with open(path, "w") as f:
        f.write("".join(header_head))
        f.write(content)
        f.write("".join(header_foot))


def generate_reference_fmha_kernels(path, num_splits=2):

    SUPPORTED_DTYPES = [
        Float,
        Half,
        BFloat,
    ]

    kernels = []

    dispatcher_fwd = DTypeDispatcher(is_backward=False)
    dispatcher_bwd = DTypeDispatcher(is_backward=True)

    for dtype_dispatcher in [dispatcher_fwd, dispatcher_bwd]:
        for dtype in SUPPORTED_DTYPES:
            dtype_dispatcher.append(dtype)
            for is_causal in [True, False]:
                kernels.append(
                    dtype_dispatcher.get_kernel_instance(
                        dtype=dtype, is_causal=is_causal
                    )
                )

    path_to_sources = f"{path}/autogen/src/cuda/reference_fmha/"
    rel_header = "natten_autogen/cuda/reference_fmha/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_api = f"{path_to_header_dir}interface.h"

    rel_path_headers = f"{rel_header}kernels.h"

    disp = dispatcher_fwd.get_dispatcher() + dispatcher_bwd.get_dispatcher()

    headers = ""
    for kernel in kernels:
        headers += kernel.get_decl()

    assert (
        len(kernels) >= num_splits
    ), f"Generated {len(kernels)} kernels, but got {num_splits=}."
    split_size = len(kernels) // num_splits
    num_splits_with_res = len(kernels) % num_splits
    kernels_emitted = []
    kernels_split = []
    for split_idx in range(num_splits):
        kernel_start_idx = split_size * split_idx + min(num_splits_with_res, split_idx)
        num_kernels_in_split = split_size + (
            1 if split_idx < num_splits_with_res else 0
        )
        kernel_end_idx = kernel_start_idx + num_kernels_in_split
        assert kernel_end_idx <= len(kernels)
        pth_set = set()
        source_list = []
        for kernel_idx in range(kernel_start_idx, kernel_end_idx):
            kernel = kernels[kernel_idx]
            source_list.append(kernel)
            kernels_emitted.append(kernel_idx)
        pth_set.add(rel_path_headers)
        write_combined_source_file(
            path_to_sources, f"source_{split_idx}.cu", sorted(pth_set), source_list
        )
        kernels_split.append(source_list)
        # print(f"{split_idx=}, {kernel_start_idx=}, {kernel_end_idx=}, {len(kernels_emitted)=}")
    assert split_idx == num_splits - 1, f"Expected {split_idx=} == {num_splits=} - 1"
    assert len(kernels_emitted) == len(kernels)
    assert sorted(kernels_emitted) == [
        x for x in range(len(kernels))
    ], f"{sorted(kernels_emitted)=}"
    assert all(len(x) > 0 for x in kernels_split)

    namespaces = ["natten", "cuda", "reference"]
    cuda_headers = [
        "natten/natten.h",
        "ATen/ATen.h",
        "ATen/cuda/CUDAContext.h",
        "c10/cuda/CUDAGuard.h",
        "c10/cuda/CUDAStream.h",
        "torch/extension.h",
        "natten/natten.h",
        "natten/helpers.h",
        "natten/cuda/reference/fmha_reference_forward.hpp",
        "natten/cuda/reference/fmha_reference_backward.hpp",
    ]
    write_header_file(disp, path_api, namespaces, cuda_headers + [rel_path_headers])
    write_header_file(headers, path_headers, namespaces, cuda_headers)


def generate_reference_fmha(output_directory: str, num_splits: int):
    generate_reference_fmha_kernels(output_directory, num_splits=num_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-directory",
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the directory where the auto-generated "
        "kernel instantiations are dumped. "
        f"Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=2,
        help="Number of source files into which the kernels are split. Default: 2.",
    )
    args = parser.parse_args()
    generate_reference_fmha(args.output_directory, args.num_splits)
