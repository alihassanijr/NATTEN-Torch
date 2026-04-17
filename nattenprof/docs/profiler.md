# nattenprof

Profiling toolkit for NATTEN. Profile neighborhood attention, standard attention, and torch
SDPA baselines with accurate per-kernel runtime breakdowns.

Uses the [PyTorch profiler API](https://docs.pytorch.org/docs/stable/profiler.html) to
capture kernel traces, maps symbols to human-readable operation names, and reports results
in pretty-printed tables or structured JSON.

## Getting Started

```bash
python -m nattenprof <subcommand> [options]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `na` | Neighborhood attention (1D, 2D, 3D) |
| `attn` | NATTEN standard attention (FMHA backends) |
| `sdpa` | PyTorch SDPA baseline (cuDNN, Flash Attention v2, xformers) |
| `batch` | Batch mode: run multiple configs from a JSON file |

### Dependencies

Only PyTorch and NATTEN are required. For the best visual experience:

```bash
pip install rich tqdm
```

## Quick Examples

### Profile neighborhood attention

```bash
python -m nattenprof na \
    -i 8 16 24 \
    -d 128 \
    --bwd
```

### Profile with a specific window size and backend

```bash
python -m nattenprof na \
    -i 32768 \
    -d 128 \
    -w 2048 \
    --dtype bf16 \
    --backend hopper-fna \
    --bwd
```

### Profile standard attention

```bash
python -m nattenprof attn \
    -q 65536 \
    -d 128 \
    --dtype bf16 \
    --backend hopper-fmha \
    --bwd
```

### Profile cuDNN SDPA baseline

```bash
python -m nattenprof sdpa \
    -q 32768 \
    -d 128 \
    --dtype bf16 \
    --backend cudnn \
    --bwd
```

### JSON output for CI / regression tracking

```bash
python -m nattenprof na \
    -i 1024 -d 128 -w 256 \
    --dtype bf16 --backend hopper-fna \
    --output-json results.json
```

## Arguments

### Shared arguments (all subcommands)

| Option | Description |
|--------|-------------|
| `-b`, `--batch-size` | QKV batch size. Default: `1`. |
| `-n`, `--heads` | Number of Q heads. Default: `1`. |
| `--heads-kv` | Number of KV heads (GQA/MQA). Defaults to `--heads`. |
| `-d`, `--dim` | QK head dim. Default: `64`. |
| `--dim-value` | V head dim if different from QK (MLA). Defaults to `--dim`. |
| `--dtype` | Element type: `fp32`, `bf16`, `fp16`, `e4m3`, `e5m2`. Default: `fp16`. |
| `--bwd` | Profile backward pass as well as forward pass. |
| `--warmup-steps` | Number of warmup iterations. Default: `10`. |
| `--output-json` | Write structured JSON results to this file path. |
| `--symbols` | Include raw kernel symbol names in output (table + JSON). |
| `--device` | CUDA GPU device index. Default: `0`. |
| `--deterministic` | Enable `torch.use_deterministic_algorithms(True)`. |
| `--init-mode` | Tensor initialization: `randn`, `uniform`, `ones`. Default: `randn`. |
| `--memory-limit` | Max GPU memory (GB) for tensor pool. Default: `10.0`. |
| `--seed` | RNG seed for tensor generation. Default: `42`. |

### `na` subcommand

| Option | Description |
|--------|-------------|
| `-i`, `--input-size` | **Required.** Token layout shape (1-3 ints). |
| `-w`, `--window-size` | Neighborhood window size. Defaults to `--input-size` (self attention). |
| `-s`, `--stride` | Stride. Defaults to `1`s. |
| `--dilation` | Dilation. Defaults to `1`s. |
| `-c`, `--causal` | Causal mask per dimension. Defaults to `False`s. |
| `--backend` | FNA backend: `cutlass-fna`, `hopper-fna`, `blackwell-fna`, `flex-fna`. |
| `--fmha-backend` | FMHA backend for self-attn fast path / cross-attn. |
| `--add-kv` | Number of additional KV tokens. Default: `0`. |
| `--q-tile` | Q tile shape in forward pass kernel. |
| `--kv-tile` | KV tile shape in forward pass kernel. |
| `--backward-q-tile` | Q tile shape in backward pass kernel. |
| `--backward-kv-tile` | KV tile shape in backward pass kernel. |
| `--schedule` | Kernel schedule (hopper only): `non`, `coop`, `pp`. |
| `--persistent` | Use persistent scheduling (blackwell only). |
| `--compile` | `torch.compile` flex attention mask + kernel. |
| `--dry-run` | Display valid configurations and exit. |
| `--optimize` | Search for the best configuration. |
| `--max-configs` | Max configs to display in dry-run. Default: `10`. `0` = show all. |
| `--optimize-warmup-steps` | Warmup steps for optimize search. Default: `5`. |

### `attn` subcommand

| Option | Description |
|--------|-------------|
| `-q`, `--seqlen` | **Required.** Q sequence length. |
| `-k`, `--seqlen-kv` | KV sequence length. Defaults to `--seqlen`. |
| `--is-causal` | Enable causal mask. |
| `--varlen` | Variable-length mode. |
| `--seqlens` | Per-batch Q sequence lengths (requires `--varlen`). |
| `--seqlens-kv` | Per-batch KV sequence lengths (requires `--varlen`). |
| `--backend` | FMHA backend: `cutlass-fmha`, `hopper-fmha`, `blackwell-fmha`, `flex-fmha`. |
| `--q-tile` | Q tile size. |
| `--kv-tile` | KV tile size. |
| `--backward-q-tile` | Backward Q tile size. |
| `--backward-kv-tile` | Backward KV tile size. |
| `--schedule` | Kernel schedule (hopper only). |
| `--persistent` | Persistent scheduling (blackwell only). |
| `--compile` | `torch.compile` (flex only). |
| `--dry-run` | Display valid configurations and exit. |
| `--optimize` | Search for the best configuration. |

### `sdpa` subcommand

| Option | Description |
|--------|-------------|
| `-q`, `--seqlen` | **Required.** Q sequence length. |
| `-k`, `--seqlen-kv` | KV sequence length. Defaults to `--seqlen`. |
| `--is-causal` | Enable causal mask. |
| `--backend` | SDPA backend: `cudnn`, `fav2`, `xformers`. Default: `cudnn`. |

### `batch` subcommand

| Option | Description |
|--------|-------------|
| `--input` | **Required.** Path to JSON input config file. |
| `--output` | **Required.** Path to JSON output results file. |
| `--print` | Also print tables to terminal while running. |


## Dry Run

List available tile shapes / configurations for your use case:

```bash
python -m nattenprof na \
    --dry-run \
    --dtype bf16 \
    -i 16 16 16 \
    -d 128
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/dry-run-h100.txt"
    ```

## Optimize

Search for the best backend configuration by profiling all available tile shapes:

```bash
python -m nattenprof na \
    -i 32768 \
    -d 128 \
    -w 2048 \
    --dtype bf16 \
    --backend hopper-fna \
    --optimize
```


## Hopper Examples

All examples include `--bwd` (forward + backward pass profiling).

### 1-D use case (32K sequence)

**Baseline (cuDNN):**

```bash
python -m nattenprof sdpa \
    -q 32768 \
    -d 128 \
    --dtype bf16 \
    --backend cudnn \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/1d-32k-cudnn-h100.txt"
    ```

**Neighborhood attention (Hopper FNA) w/ 2K sliding window:**

```bash
python -m nattenprof na \
    -i 32768 \
    -d 128 \
    -w 2048 \
    --dtype bf16 \
    --backend hopper-fna \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/1d-32k-w2k-hopper-fna-h100.txt"
    ```

**Strided NA w/ stride 256:**

```bash
python -m nattenprof na \
    -i 32768 \
    -d 128 \
    -w 2048 \
    -s 256 \
    --dtype bf16 \
    --backend hopper-fna \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/1d-32k-w2k-s256-hopper-fna-h100.txt"
    ```

**Blocked attention (stride == window size):**

```bash
python -m nattenprof na \
    -i 32768 \
    -d 128 \
    -w 2048 \
    -s 2048 \
    --dtype bf16 \
    --backend hopper-fna \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/1d-32k-w2k-s2k-hopper-fna-h100.txt"
    ```


### 2-D use case (FLUX)

256x256 token layout, 24 heads, head dim 128.

**Baseline (cuDNN):**

```bash
python -m nattenprof sdpa \
    -q 65536 \
    -n 24 \
    -d 128 \
    --dtype bf16 \
    --backend cudnn \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/2d-flux-cudnn-h100.txt"
    ```

**Neighborhood attention w=(80,80):**

```bash
python -m nattenprof na \
    -i 256 256 \
    -n 24 \
    -d 128 \
    -w 80 80 \
    --dtype bf16 \
    --backend hopper-fna \
    --q-tile 16 8 --kv-tile 16 8 \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/2d-flux-na-hopper-fna-h100.txt"
    ```

**GNA w=(80,80) s=(16,16):**

```bash
python -m nattenprof na \
    -i 256 256 \
    -n 24 \
    -d 128 \
    -w 80 80 \
    -s 16 16 \
    --dtype bf16 \
    --backend hopper-fna \
    --q-tile 16 8 --kv-tile 16 8 \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/2d-flux-gna-hopper-fna-h100.txt"
    ```


### 3-D use case (Hunyuan Video)

30x48x80 token layout, 24 heads, head dim 128.

**Baseline (cuDNN):**

```bash
python -m nattenprof sdpa \
    -q 115200 \
    -n 24 \
    -d 128 \
    --dtype bf16 \
    --backend cudnn \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/3d-hunyuan-cudnn-h100.txt"
    ```

**Neighborhood attention w=(18,24,24):**

```bash
python -m nattenprof na \
    -i 30 48 80 \
    -n 24 \
    -d 128 \
    -w 18 24 24 \
    --dtype bf16 \
    --backend hopper-fna \
    --q-tile 2 8 8 --kv-tile 2 8 8 \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/3d-hunyuan-na-hopper-fna-h100.txt"
    ```

**GNA w=(18,24,24) s=(16,8,8):**

```bash
python -m nattenprof na \
    -i 30 48 80 \
    -n 24 \
    -d 128 \
    -w 18 24 24 \
    -s 16 8 8 \
    --dtype bf16 \
    --backend hopper-fna \
    --q-tile 2 8 8 --kv-tile 2 8 8 \
    --bwd
```

??? hopper-example "Sample output from running on H100"
    ```
    --8<-- "nattenprof/docs/sample-outputs/3d-hunyuan-gna-hopper-fna-h100.txt"
    ```


## Blackwell Examples

!!! warning "TODO"
    Blackwell sample outputs not yet generated. Run `nattenprof/docs/generate-samples.sh`
    on a Blackwell cluster to populate them. See `nattenprof/docs/BLACKWELL_TODO.md`.


## Batch Mode

Run multiple configurations from a JSON file:

```bash
python -m nattenprof batch \
    --input configs.json \
    --output results.json \
    --print
```

### Input format

```json
[
    {
        "op": "na",
        "input_size": [256, 256],
        "window_size": [80, 80],
        "stride": [16, 16],
        "dim": 128,
        "heads": 24,
        "dtype": "bf16",
        "backend": "hopper-fna",
        "q_tile": [16, 8],
        "kv_tile": [16, 8]
    },
    {
        "op": "attn",
        "seqlen": 65536,
        "dim": 128,
        "heads": 24,
        "dtype": "bf16",
        "backend": "hopper-fmha"
    },
    {
        "op": "sdpa",
        "seqlen": 65536,
        "dim": 128,
        "heads": 24,
        "dtype": "bf16",
        "backend": "cudnn"
    }
]
```

### Output format

JSON output includes metadata, per-kernel breakdowns, and timing:

```json
{
    "metadata": {
        "timestamp": "2026-04-15T12:00:00",
        "gpu": "NVIDIA H100 80GB HBM3",
        "torch_version": "2.11.0+cu129",
        "natten_version": "0.21.6"
    },
    "results": [
        {
            "operation": "na",
            "config": { ... },
            "total_us": 10601.0,
            "forward_us": 10601.0,
            "backward_us": 0.0,
            "breakdown": {
                "attention_us": 7914.0,
                "token_permute_us": 2687.0,
                "reduction_us": 0.0,
                "elementwise_us": 0.0,
                "other_us": 0.0
            },
            "kernels": [ ... ]
        }
    ]
}
```

## Tensor Pool

nattenprof allocates multiple copies of input tensors and cycles through them during
profiling. This avoids L2 cache reuse artifacts that can skew results when the same tensors
are used every iteration.

The pool size is automatically computed to fit within `--memory-limit` (default 10 GB),
with a minimum of 2 copies and maximum of 256.

| Option | Description |
|--------|-------------|
| `--init-mode` | Tensor initialization: `randn` (default), `uniform`, `ones`. |
| `--memory-limit` | Max GPU memory (GB) for tensor pool. Default: `10.0`. |
| `--seed` | RNG seed for deterministic tensor generation. Default: `42`. |
