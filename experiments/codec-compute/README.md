# Codec computational remapping

Can dedicated media engines take on operations that normally run on the CPU? The experiments progressed from a software PNG proof to exact addition on NVIDIA's NVDEC hardware and comparisons against native AVX2.

[Experiment log, benchmark results, and reproduction commands](RESEARCH.md)

## Run the quick checks

The software proof requires Python 3.10+ and FFmpeg. It builds a grayscale PNG whose Up filter reconstructs 4,096 bytewise sums modulo 256.

```sh
python3 experiments/codec-compute/run.py --mode cpu --out cpu-results.json
```

The optional hardware availability probe generates synthetic H.264 video and explicitly selects the NVIDIA decoder. It requires an NVIDIA driver, `nvidia-smi`, and FFmpeg with `h264_cuvid` and `libx264`. The probe establishes video decode support. The [arithmetic benchmarks](RESEARCH.md#reproduce-the-experiments) verify mathematical results separately.

```sh
python3 experiments/codec-compute/run.py --mode nvdec --out nvdec-results.json
```

## GitHub Actions

PRs run the software proof on a standard Ubuntu runner. The workflow saves diagnostics and JSON artifacts and maintains one automatically updated PR comment. A failure still returns a failed check. The optional NVDEC workflow requires a Linux x64 self-hosted runner labeled `nvdec`. In GitHub Actions, dispatch **Codec compute remapping** with `run_nvdec` after that runner is available. The archived hardware results came from a free Colab Tesla T4 runtime.

The [archived results](results/) and [benchmark sources](benchmarks/) use synthetic inputs.
