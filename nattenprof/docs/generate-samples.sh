#!/bin/bash
# Generate sample outputs for nattenprof docs.
# Run on the target GPU cluster (Hopper or Blackwell).
# All profiling runs include --bwd (forward + backward).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
# TODO(relocate): Remove this PYTHONPATH export once nattenprof lives under
# src/natten/profiler/. It is only needed while nattenprof sits at repo root.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"

OUTDIR="$SCRIPT_DIR/sample-outputs"
mkdir -p "$OUTDIR"

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
echo "GPU: $GPU_NAME"

if echo "$GPU_NAME" | grep -qi "h100"; then
    TAG="h100"
elif echo "$GPU_NAME" | grep -qi "b200\|b100\|blackwell"; then
    TAG="b200"
else
    TAG="unknown"
fi

echo "Tag: $TAG"

run() {
    local name="$1"
    shift
    local outfile="$OUTDIR/${name}-${TAG}.txt"
    echo "=== Generating $outfile ==="
    python -m nattenprof "$@" 2>&1 | grep -v "UserWarning\|_subclasses\|_warn_once\|Profiler clears\|_conversion_method_template" > "$outfile"
}

# --- 1D: 32K sequence ---

# Self attention baseline (cuDNN)
run "1d-32k-cudnn" sdpa -q 32768 -d 128 --dtype bf16 --backend cudnn --bwd

if [ "$TAG" = "h100" ]; then
    FNA_BACKEND="hopper-fna"
    FMHA_BACKEND="hopper-fmha"

    # NA sliding window w=2048
    run "1d-32k-w2k-hopper-fna" na -i 32768 -d 128 -w 2048 --dtype bf16 --backend $FNA_BACKEND --bwd

    # Strided NA w=2048 s=256
    run "1d-32k-w2k-s256-hopper-fna" na -i 32768 -d 128 -w 2048 -s 256 --dtype bf16 --backend $FNA_BACKEND --bwd

    # Blocked attention w=2048 s=2048
    run "1d-32k-w2k-s2k-hopper-fna" na -i 32768 -d 128 -w 2048 -s 2048 --dtype bf16 --backend $FNA_BACKEND --bwd

    # --- 2D: FLUX (256x256, 24 heads) ---

    run "2d-flux-cudnn" sdpa -q 65536 -n 24 -d 128 --dtype bf16 --backend cudnn --bwd

    # Hopper 2D tiles: fwd q=(16,8) kv=(16,8), bwd same
    run "2d-flux-na-hopper-fna" na -i 256 256 -n 24 -d 128 -w 80 80 --dtype bf16 --backend $FNA_BACKEND --q-tile 16 8 --kv-tile 16 8 --backward-q-tile 16 8 --backward-kv-tile 16 8 --bwd
    run "2d-flux-gna-hopper-fna" na -i 256 256 -n 24 -d 128 -w 80 80 -s 16 16 --dtype bf16 --backend $FNA_BACKEND --q-tile 16 8 --kv-tile 16 8 --backward-q-tile 16 8 --backward-kv-tile 16 8 --bwd

    # --- 3D: Hunyuan (30x48x80, 24 heads) ---

    run "3d-hunyuan-cudnn" sdpa -q 115200 -n 24 -d 128 --dtype bf16 --backend cudnn --bwd

    # Hopper 3D tiles: fwd q=(2,8,8) kv=(2,8,8), bwd same
    run "3d-hunyuan-na-hopper-fna" na -i 30 48 80 -n 24 -d 128 -w 18 24 24 --dtype bf16 --backend $FNA_BACKEND --q-tile 2 8 8 --kv-tile 2 8 8 --backward-q-tile 2 8 8 --backward-kv-tile 2 8 8 --bwd
    run "3d-hunyuan-gna-hopper-fna" na -i 30 48 80 -n 24 -d 128 -w 18 24 24 -s 16 8 8 --dtype bf16 --backend $FNA_BACKEND --q-tile 2 8 8 --kv-tile 2 8 8 --backward-q-tile 2 8 8 --backward-kv-tile 2 8 8 --bwd

    # --- NATTEN attention (FMHA) ---
    run "attn-hopper-fmha" attn -q 1024 -d 128 --dtype bf16 --backend $FMHA_BACKEND --bwd

elif [ "$TAG" = "b200" ]; then
    FNA_BACKEND="blackwell-fna"
    FMHA_BACKEND="blackwell-fmha"

    # NA sliding window w=2048
    run "1d-32k-w2k-blackwell-fna" na -i 32768 -d 128 -w 2048 --dtype bf16 --backend $FNA_BACKEND --bwd

    # Strided NA w=2048 s=256
    run "1d-32k-w2k-s256-blackwell-fna" na -i 32768 -d 128 -w 2048 -s 256 --dtype bf16 --backend $FNA_BACKEND --bwd

    # Blocked attention w=2048 s=2048
    run "1d-32k-w2k-s2k-blackwell-fna" na -i 32768 -d 128 -w 2048 -s 2048 --dtype bf16 --backend $FNA_BACKEND --bwd

    # --- 2D: FLUX (256x256, 24 heads) ---

    run "2d-flux-cudnn" sdpa -q 65536 -n 24 -d 128 --dtype bf16 --backend cudnn --bwd

    # Blackwell 2D tiles: fwd q=(16,16) kv=(16,8), bwd q=kv=(16,8)
    run "2d-flux-na-blackwell-fna" na -i 256 256 -n 24 -d 128 -w 80 80 --dtype bf16 --backend $FNA_BACKEND --q-tile 16 16 --kv-tile 16 8 --backward-q-tile 16 8 --backward-kv-tile 16 8 --bwd
    run "2d-flux-gna-blackwell-fna" na -i 256 256 -n 24 -d 128 -w 80 80 -s 16 16 --dtype bf16 --backend $FNA_BACKEND --q-tile 16 16 --kv-tile 16 8 --backward-q-tile 16 8 --backward-kv-tile 16 8 --bwd

    # --- 3D: Hunyuan (30x48x80, 24 heads) ---

    run "3d-hunyuan-cudnn" sdpa -q 115200 -n 24 -d 128 --dtype bf16 --backend cudnn --bwd

    # Blackwell 3D tiles: fwd q=(4,8,8) kv=(2,8,8), bwd q=kv=(2,8,8)
    run "3d-hunyuan-na-blackwell-fna" na -i 30 48 80 -n 24 -d 128 -w 18 24 24 --dtype bf16 --backend $FNA_BACKEND --q-tile 4 8 8 --kv-tile 2 8 8 --backward-q-tile 2 8 8 --backward-kv-tile 2 8 8 --bwd
    run "3d-hunyuan-gna-blackwell-fna" na -i 30 48 80 -n 24 -d 128 -w 18 24 24 -s 16 8 8 --dtype bf16 --backend $FNA_BACKEND --q-tile 4 8 8 --kv-tile 2 8 8 --backward-q-tile 2 8 8 --backward-kv-tile 2 8 8 --bwd

    # --- NATTEN attention (FMHA) ---
    run "attn-blackwell-fmha" attn -q 1024 -d 128 --dtype bf16 --backend $FMHA_BACKEND --bwd

else
    echo "WARNING: Unknown GPU tag '$TAG'. Skipping backend-specific runs."
fi

# --- Dry run ---
run "dry-run" na -i 16 16 16 -d 128 --dtype bf16 --dry-run

# --- NATTEN attention (cutlass-fmha, works on all arches) ---
run "attn-cutlass-fmha-causal" attn -q 1024 -d 128 --dtype bf16 --backend cutlass-fmha --is-causal --bwd

echo "=== Done. Samples in $OUTDIR ==="
ls -la "$OUTDIR"/*-${TAG}.txt
