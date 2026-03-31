# Testing NATTEN

## Commands

When running the entire test suite:

```bash
# Sequential (stops on first failure)
make test

# Parallel across GPUs (requires GNU parallel)
make test_parallel GPUS=4
make test_parallel GPUS=4 WORKERS=8
```

Parallel runs write per-worker logs to `test-logs/`.

## Individual Test Files

| File                    | What it tests |
|-------------------------|-----------------------------------------------------------------------------------|
| `test_fmha.py`          | All FMHA backends (cutlass, hopper, blackwell)                                    |
| `test_fna.py`           | CUTLASS-FNA backend                                                               |
| `test_hopper_fna.py`    | Hopper FNA backend                                                                |
| `test_blackwell_fna.py` | Blackwell FNA backend                                                             |
| `test_fmha_varlen.py`   | Variable-length attention feature                                                 |
| `test_flex.py`          | Flex Attention backend (FMHA and FNA)                                             |
| `test_token_permute.py` | Token permutation ops (used in Hopper FNA, Blackwell FNA, and sometimes Flex FNA) |
| `test_compute_delta.py` | Compute delta kernel (used only in CUTLASS-FNA/FMHA)                              |
| `test_attn_merge.py`    | Attention merging operation                                                       |
| `test_torch_compile.py` | `torch.compile` compatibility                                                     |

Architecture-specific tests auto-skip on unsupported GPUs.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NATTEN_LOG_LEVEL` | `CRITICAL` (in `make test`) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `NATTEN_RUN_EXTENDED_TESTS` | unset | `1` to enable larger parameter sweeps |
| `NATTEN_RAND_SWEEP_TESTS` | unset | Number of random test cases (e.g. `100`, `1000`) |
| `NATTEN_RUN_FLEX_TESTS` | unset | `0` to skip flex tests (slow) |
| `NATTEN_TOKPERM_DEFAULT_IMPL` | `cutlass` | `cutlass` or `torch` for token permute tests |


## Running Specific Tests

```bash
pytest -v tests/test_fna.py::FNABackendTest::test_2d_against_reference
pytest -v -k "hopper and 2d"
```
