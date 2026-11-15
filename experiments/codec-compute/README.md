# Codec computational remapping

Can dedicated media engines take on operations that normally run on the CPU while CUDA is busy? The verified compute route encodes operands and H.264 prediction instructions, lets NVDEC reconstruct exact arithmetic results, then uses NVENC to carry results into another stage. The experiments progressed from a software PNG proof to exact addition on NVIDIA's NVDEC hardware, codec feedback chains, comparisons against native AVX2, and an XOR bit-plane mapping.

The [pipeline roles and next experiment](ARCHITECTURE.md) keep NVDEC and NVENC as the arithmetic route. A security engine, if exposed to applications on the test device, could support that route with a separately needed hash, authentication, or encryption task. The security XOR software proof is an exploratory mapping, not a replacement for the verified codec route. Neither a T4 security-hardware result nor a speedup with concurrent CUDA work has been demonstrated.

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
