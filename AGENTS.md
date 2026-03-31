# AGENTS.md
Guide for coding agents.

## Project Overview

NATTEN ships CUDA kernels and other implementations of the neighborhood attention family of
patterns: sliding window, strided sliding window, blocked, dilated window, and the like.
It focuses specifically on multi-dimensional token layouts.

Nearly all compute CUDA kernels are built with NVIDIA CUTLASS, and other kernels more or less depend
on utilities from CUTLASS.

Frontend is all PyTorch: tensors in, tensors out, with hierarchical verification and checks
preventing incorrect behavior.
End-product should always integrate as seamlessly as possible with the simplest torch programs.

Project is minimal dependency by design: only Python, PyTorch and CUDA driver and runtime are
required for running, and only cmake and relevant compiler toolkits are required for building.
Development / debugging can require additional dependencies.
Certain parts like the profiler will (visually) benefit from some additional dependencies like
`tqdm` and `rich`, but they are optional.

More documentation is available under `docs/`, and in docstrings under `src/`.

Refer to [.agents/TERMINOLOGY.md](.agents/TERMINOLOGY.md) as needed for finding abbreviations.

Refer to [.agents/CODEORG.md](.agents/CODEORG.md) for code organization.

Refer to [.agents/CUTLASS.md](.agents/CUTLASS.md) for information on CUTLASS.

