# Codec computational remapping experiment

**Question:** Can ordinary CPU computations be encoded into data structures that cause a codec to perform the operation during reconstruction?

## CPU proof (runs automatically on public GitHub Actions)

Requires Python 3.10+ and FFmpeg. No third-party Python packages.

    python3 experiments/codec-compute/run.py --mode cpu --out cpu-results.json

A valid grayscale PNG is built with row 1 containing A and row 2 using the PNG **Up** filter with encoded values B. The PNG decoder must reconstruct row 2 as **(A + B) modulo 256**. The test compares all 4,096 output bytes against the expected result and fails on mismatch.

**Important:** FFmpeg decodes this PNG using CPU software. It proves mathematical remapping, not GPU acceleration. Its timing includes process startup, image processing, and I/O; it does not establish a speed advantage over an optimized native CPU operation.

## NVIDIA hardware probe (optional)

Requires NVIDIA drivers, working `nvidia-smi`, and an FFmpeg build with `h264_cuvid` and `libx264`:

    python3 experiments/codec-compute/run.py --mode nvdec --out nvdec-results.json

This generates synthetic H.264 video, then compares software H.264 decoding to **explicit** NVDEC decoding. Missing NVIDIA hardware or decoding support produces an actual failure diagnostic. No CPU fallback is permitted.

**Important:** This hardware probe establishes video decoding availability only. It does **not** demonstrate general arithmetic on NVDEC.

## GitHub workflow and cost

Pull requests automatically run the CPU proof on standard hosted Ubuntu machines, which are free for public repos. One automatically updated PR comment contains a result, real errors, and a workflow link; failed steps remain failed. Raw logs and JSON results are artifacts.

To run the hardware probe, first register your own Linux x64 NVIDIA GPU machine as a GitHub self-hosted runner and add the `nvdec` label. Then manually dispatch **Codec compute remapping** with the `run_nvdec` option. **Do not enable the hardware option without a matching runner; the job would remain queued.** Standard free GitHub runners do not include GPU hardware. External GPU notebook services are an alternative, but CUDA access does not guarantee NVDEC access.

## Next unproven research step

Construct a valid H.264 reference prediction/residual bitstream to implement a specified **non-video numerical operation inside NVDEC**. Verify exactness; measure preparation overhead, CPU load, memory transfer, latency, throughput, and energy against optimized CPU and programmable GPU equivalents. This has **not** yet been implemented or validated.

No monkey patches, simulated hardware claims, proprietary datasets, or undocumented stub implementations are used.
