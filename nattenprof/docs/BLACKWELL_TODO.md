# Blackwell sample outputs

DONE: `generate-samples.sh` updated with Blackwell-specific backends and tile shapes,
and `*-b200.txt` files generated on GB200 (SM100).

## Re-generating

1. Get on a node with B200 GPU(s)
2. Activate the venv: `source .venv/bin/activate`
3. Run: `bash nattenprof/docs/generate-samples.sh`

The script auto-detects the GPU name and tags output files accordingly (`-b200.txt`).
Same script works on both Hopper and Blackwell.

## What gets generated

The script produces sample outputs for:
- 1D 32K sequence: cuDNN baseline, NA w=2048, strided s=256, blocked s=2048
- 2D FLUX (256x256, 24 heads): cuDNN baseline, NA w=(80,80), GNA w=(80,80) s=(16,16)
- 3D Hunyuan (30x48x80, 24 heads): cuDNN baseline, NA w=(18,24,24), GNA s=(16,8,8)
- Dry run (3D)
- NATTEN attention (FMHA): blackwell-fmha / hopper-fmha, cutlass-fmha causal

## Blackwell tile shapes (from docs/profiler.md)

- 2D: fwd q=(16,16) kv=(16,8), bwd q=kv=(16,8)
- 3D: fwd q=(4,8,8) kv=(2,8,8), bwd q=kv=(2,8,8)
