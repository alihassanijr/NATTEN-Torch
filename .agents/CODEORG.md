# Code organization

1. **libnatten** (`csrc/`): C++ and CUDA layer holding all ahead-of-time-compiled kernels. Built by
   `cmake` during pip install (only if building a distribution or if user has a compatible CUDA device).

2. **Python Frontend** (`src/natten/`): PyTorch integration, libnatten loading (`_libnatten/`),
   libnatten backend management (`backends/`), utilities (`utils`), checks and verification layer,
   front-end APIs (`functional.py`, `modules.py`), and profiling toolkit (`profiler.py` and
   `profiling_utils/`).

## Attention Backends (implementations)
Various FMHA and FNA implementations (see terminology) are supported.

* `blackwell-fmha` / `blackwell-fna`: kernels for Blackwell DC-class GPUs ONLY.
* `hopper-fmha` / `hopper-fna`: kernels for Hopper GPUs ONLY.
* `cutlass-fmha` / `cutlass-fna`: multi-arch kernels for SM50, SM70, SM75, and SM80 (see terminology).
* `flex-fmha` / `flex-fna`: based on PyTorch's Flex Attention API (just-in-time, limited support,
    experimental, not tied to CUDA, not tied to libnatten). See `backends/flex.py`.

## End-user APIs

* Primary operators (`natten.functional`)
    * `attention`: standard dot-product attention
    * `neighborhood_attention_generic`: neighborhood attention

* Secondary operators (`natten.functional`)
    * `merge_attentions`: merges attention outputs corresponding to the same queries over different
        splits of key-value pairs.

* Memory operator (`natten.token_permute`):
    * Some FNA backends (`hopper-fna`, `blackwell-fna`) use memory operations to handle
        multi-dimensional tiling.
    * `token_permute_operation`: permute tokens for a decomposed FNA kernel
    * `token_unpermute_operation`: revert permuted tokens back into original layout after decomposed FNA kernel.

* Modules (`NeighborhoodAttention{1,2,3}D`): torch nn modules for easy integration into torch
    models. Modules mostly serve as examples of how to use primary operators.
