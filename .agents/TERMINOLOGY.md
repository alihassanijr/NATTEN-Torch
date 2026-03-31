# TERMINOLOGY.md

List of terms and abbreviations used throughout the projects.

* **NA**: Neighborhood Attention is a family of sparse attention methods (dot-product sparsity) that
    implement sliding window (`window_size`/`kernel_size`), strided sliding window, dilated sliding
    window, and causal sliding window attention in multiple dimensions. Various forms of NA can
    always be defined as attention masks.
    Neighborhood Attention is strictly defined over **Self-Attention** problems: query and key-value
    pair (context) live in the same token coordinate space / token layout. It is not defined for
    cross-attention.

* **FMHA**: Fused multi-headed attention refers to implementations such as Flash Attention that fuse
    attention BMMs into a single kernel (2 BMMs in forward pass, 5 BMMs in backward pass). This
    fusion avoids having to materialize the dot-products in global memory, eliminating the quadratic
    memory footprint of **Self-Attention** (not cross-attention), and breaks the global memory
    bandwidth wall.

* **FNA**: Fused Neighborhood Attention kernels are counterparts to FMHA kernels that implement
    neighborhood attention. 3 key differences between FNA and FMHA kernels:
        1. Multi-dimensional tiling (preserves spatial locality, better throughput)
        2. Block-sparse scheduling (made much simpler by #1)
        3. NA fine-grained mask (numerical correctness)

* **TokPerm**: token permute / token permutation is a memory operation that performs
    multi-dimensional tiling (#1 in FNA) outside the main compute kernel, in order to simplify the
    design, improve development speed, and in some cases even improve the final performance.

* **Blackwell DC-class**": Blackwell Data Center class GPU: either the B200/GB200 (SM100) or
    B300/GB300 (SM103). Kernels using PTX features for this specific family is not forward
    compatible.

* `SM[0-9]{2,3}`: indicates CUDA GPU architecture / compute capability.
    * SM50 (compute capability 5.0): Maxwell (deprecated)
    * SM60: Pascal (deprecated)
    * SM70: Volta
    * SM75: Turing
    * SM80: Ampere DC-class (i.e. A100)
    * SM86: Ampere RTX
    * SM89: Ada (RTX only)
    * SM90: Hopper (DC-class only)
    * SM100: Blackwell (DC-class)
    * SM103: "Blackwell Ultra" (DC-class)
    * SM120 / SM121: Blackwell RTX / consumer

* Generic kernels: Kernels in pure CUDA C++ are usually compilable on all architectures.
* Forward-compatible kernels: SM80 kernels compile and run on all architectures since Ampere.
* Kernels that are not forward-compatible: Hopper and Blackwell DC-class have architecture-specific instructions and features, which are enabled by adding `a` to arch tag when compiling (i.e. `sm90a` / `sm100a`).
