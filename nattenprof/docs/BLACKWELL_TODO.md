# Blackwell sample outputs TODO

Run `generate-samples.sh` on a Blackwell (B200) cluster to populate the `*-b200.txt` files.

## Instructions

1. Get on a node with B200 GPU(s)
2. Activate the venv: `source .venv/bin/activate`
3. Run: `bash nattenprof/docs/generate-samples.sh`

The script auto-detects the GPU name and tags output files accordingly (`-b200.txt`).
No code changes needed — same script works on both Hopper and Blackwell.

## What gets generated

The script produces sample outputs for:
- 1D 32K sequence: cuDNN baseline, NA w=2048, NA w=2048 +bwd, strided, blocked
- 2D FLUX (256x256, 24 heads): cuDNN baseline, NA w=(80,80), NA +bwd, GNA w=(80,80) s=(16,16)
- 3D Hunyuan (30x48x80, 24 heads): cuDNN baseline, NA w=(18,24,24), NA +bwd, GNA s=(16,8,8)
- Dry run (3D)
- NATTEN attention (FMHA): hopper-fmha, cutlass-fmha causal

On Blackwell, the script will automatically use `blackwell-fna` / `blackwell-fmha` if you
update the commands. Currently it uses `hopper-fna` — you may want to add Blackwell-specific
runs to the script, or duplicate the script with Blackwell backends.

For Blackwell-specific tile sizes (e.g. Q tile 256, KV tile 128), add additional runs to
`generate-samples.sh`.
