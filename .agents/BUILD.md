# Building NATTEN

## Prerequisites
- Python >= 3.9, PyTorch >= 2.5, CUDA Toolkit >= 12.0, CMake >= 4.0

## Commands

First check for a virtual environment, prompt the user if you find multiple or none.

Then activate, and ensure dependencies are met (torch must be installed).

Then build:
```bash
# Development build (editable install)
make dev

# Standard Build
make

# Build for specific architectures (i.e. SM90: Hopper)
make CUDA_ARCH="9.0"

# Build with more parallel workers
make CUDA_ARCH="9.0" WORKERS=8

# Build with max workers
make CUDA_ARCH="9.0" WORKERS=$(nproc)

# Build with max workers and max build targets (fastest)
make CUDA_ARCH="9.0" WORKERS=$(nproc) AUTOGEN_POLICY="fine"
```

## Key Environment Variables

| Variable                | Makefile target  | Default       | Description                                                           |
|-------------------------|------------------|---------------|-----------------------------------------------------------------------|
| `NATTEN_CUDA_ARCH`      | `CUDA_ARCH`      | auto-detected | Target SM arch(es), semicolon-separated (e.g. `"8.0;9.0"`)            |
| `NATTEN_AUTOGEN_POLICY` | `AUTOGEN_POLICY` | `default`     | `fine` (more parallelism), `default`, or `coarse` (fewer targets)     |
| `NATTEN_N_WORKERS`      | `WORKERS`        | `nproc/4`     | Parallel build workers                                                |
| `NATTEN_VERBOSE`        | `VERBOSE`        | `0`           | Verbose cmake/compiler output                                         |
| `NATTEN_BUILD_DIR`      | -                | system tmpdir | Build directory (makefile sets `$(PWD)/build_dir/` to avoid rebuilds) |

Debug-only:

| Variable                     | Default | Description                   |
|------------------------------|---------|-------------------------------|
| `NATTEN_BUILD_WITH_PTX`      | `0`     | Include PTX (virtual) targets |
| `NATTEN_BUILD_WITH_LINEINFO` | `0`     | Add `-lineinfo` to nvcc       |

## Autogen

`scripts/autogen_*.py` generate kernel instantiations into `csrc/autogen/`. This runs automatically
during build. **Never hand-edit files under `csrc/autogen/`.** Never try to parse or read
`csrc/autogen/` unless it is necessary (i.e. when you're making changes to the autogen scripts or
making new ones.)

Autogen splits kernel instantiations across multiple `.cu` files for build parallelism. The
`NATTEN_AUTOGEN_POLICY` controls the granularity (number of splits per kernel category).

## Build Artifacts

- `src/natten/libnatten.*.so` -- compiled C++ extension loaded at runtime
- `build_dir/` -- cmake build cache (persists across rebuilds when `NATTEN_BUILD_DIR` is set)
- `csrc/autogen/` -- generated kernel instantiation sources

`make clean` removes Python artifacts and the compiled `.so`. `make deep-clean` also removes
`build_dir/` and `csrc/autogen/`.

## Architecture Notes

SM90 and SM100/SM103 use architecture-specific instructions (`sm90a`, `sm100a`), which are
**not** forward-compatible.
See [TERMINOLOGY.md](TERMINOLOGY.md) for details.
