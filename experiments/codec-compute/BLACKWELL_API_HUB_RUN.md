# Blackwell 4000 API Hub run

Run this on the Blackwell 4000 machine in API Hub. First record the exact `nvidia-smi` product name: RTX PRO 4000 Blackwell desktop, SFF, and laptop models are separate configurations. Keep all comparisons on this one machine; the archived Tesla T4 runs are historical references, not paired speedup baselines. NVIDIA lists two NVENC and two NVDEC engines for the [desktop RTX PRO 4000 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4000/); do not assume that count for another model.

## 1. Capture capability and software inventory

From `experiments/codec-compute`, save the output of these read-only commands with the run results:

```sh
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv
nvidia-smi -q
ffmpeg -version
ffmpeg -hide_banner -encoders
ffmpeg -hide_banner -decoders
python3 --version
```

Confirm `h264_nvenc` and `h264_cuvid` are listed. Record OS, available CUDA headers, whether this is a container or remote session, and whether the user can build native code. A missing library or permission is a capability result, not a successful hardware test. Do not infer SEC2 application access from `nvidia-smi` alone.

## 2. Reproduce the codec baseline

Create a fresh output directory. Run the existing exact-output route on this GPU and keep the complete JSON and stderr:

```sh
mkdir -p results/blackwell-api-hub
python3 benchmarks/codec_xor_decode_encode.py --hardware --output results/blackwell-api-hub/xor.json
python3 benchmarks/codec_reuse_concurrency.py --hardware --pairs 512 --repeats 8 --output results/blackwell-api-hub/reuse.json
python3 benchmarks/codec_completion_chain.py --hardware --frames 128 --output results/blackwell-api-hub/handoffs.json
```

Before calling any result a CPU offload, compare equal raw inputs and usable byte outputs with a direct CPU route. Count host plus child-process CPU time, including preparation, stream packing, FFmpeg, copies, output reconstruction, and orchestration. Repeat on this machine with its existing CUDA workload active; report CUDA throughput and productive CPU-worker throughput in the same window. If API Hub does not provide an unrelated CUDA workload, mark that comparison pending.

## 3. Probe SEC2 API feasibility

Inspect the installed driver and public source for the exact GPU/driver version. Map SEC2 commands, supported buffer operations, submission rights, completion signals, and any path that could advance a ready NVDEC/NVENC job. Record the relevant source locations and actual error codes. The Jetson `ecb-aes-tegra` and `cbc-aes-tegra` probe is not evidence for desktop or laptop SEC2: it targets a different driver interface.

If a documented or verifiably supported driver command accepts user data, implement a small versioned adapter with explicit errors, exact known-answer output, and engine-activity evidence. Compare its CPU time and wall time with the same support work done on CPU, while the codec route stays in place. If a completion signal can advance a bounded codec queue, measure dispatch latency, ordering, backpressure, and process-tree CPU time. Identify whether the driver still schedules each job. Do not patch the live driver, write speculative registers, or label a CPU-hosted wrapper as SEC2 execution.

If the available firmware has no suitable command, record that gap. Creating an application API is still possible; autonomous SEC2 scheduling would then require a separately demonstrated firmware and driver capability.

## Report

For each run, preserve the command, exit code, useful stderr, environment inventory, exact-output check, wall time, total process-tree CPU time, CUDA and CPU-worker throughput when available, and whether the security engine was **verified**, **unverified**, or **not accessible**. Keep device-specific results in this directory and add a short finding to [RESEARCH.md](RESEARCH.md). Do not claim a hardware result from a software simulation.
