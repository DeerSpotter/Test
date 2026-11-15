# Codec compute experiment log

The project explores whether a GPU's fixed-function hardware can execute useful work while its programmable cores are available for other jobs. The tests use synthetic data. NVIDIA hardware runs used a Tesla T4 in Colab. The archived records include exactness checks, test inputs or their generators, and timing details.

## What was measured

| Step | Setup and observed result |
|---|---|
| [01. Software proof and hardware discovery](results/01-initial-software.json) | The PNG Up filter produced 16,384 exact modular additions. A second PNG test produced 253,952 modular running-sum additions in 0.511 ms, versus 0.00948 ms using compiled CPU code and 1.539 ms using the tested NumPy implementation. A 4 MiB software FFV1 round trip recovered all bytes. FFV1 was timed with process and file I/O, while zlib used in-memory buffers. The first NVENC attempt failed with `Cannot load libcuda.so.1`. |
| [02. T4 hardware access](results/02-colab-nvdec.json) | Explicit `h264_cuvid` decoding succeeded. A strict follow-up compared 120 CUDA-decoded H.264 frames against the CPU decoder, with every frame hash matching. The quick PNG probe also verified 4,096 software-decoded byte sums. |
| [03. One encoded input plus offsets](results/03-weighted-offset.json) | NVENC preserved the structured 256 × 256 reference frame. An H.264 P picture requested eight signed offsets. Using weighted prediction, NVDEC produced all 65,536 exact clipped sums and passed the zero-offset control. An initial 128 × 128 input was rejected by the encoder as too small. |
| [04. Parallel feedback](results/04-parallel-feedback.json) | In four chains of three rounds, the encoded bitstream selected each offset plan and the decoded bytes became the next encoder input. Serial and parallel outputs matched. Two runs per mode averaged 17.46 s serial and 9.80 s parallel, a **1.78× throughput improvement**. Host CPU time went from 14.21 s to 12.67 s. Encode and decode subprocesses overlapped by about 8.00 s. The 200 ms device counters read 0%, so exact on-chip overlap was unresolved. |
| [05. Batched offsets versus AVX2](results/05-batched-avx2.json) | A 512-frame batch with three dependent offset passes produced 100.7 million exact additions. The CPU took 34.2 ms for raw inputs versus 1,614 ms for NVENC and NVDEC. For encoded inputs, normal NVDEC decode plus AVX2 took 723 ms versus 960 ms for arithmetic inside NVDEC. Offsets were fixed up front and the intermediate prediction frames stayed in NVDEC. |
| [06. One extra frame](results/06-one-frame-avx2.json) | For one offset pass over 512 encoded frames, plain decode took 722 ms, decode plus AVX2 took 786 ms, and weighted NVDEC took 1,027 ms including preparation or 863 ms with a prepared stream. Eight of 32 offset sequences from the preceding three-pass case had an exact equivalent using one supported signed offset. |
| [07. Two independent encoded operands](results/07-two-input-addition.json) | NVENC preserved both inputs. A weighted H.264 B picture used two references, zero motion and zero residuals. NVDEC returned 65,536 exact sums across 256 distinct operand pairs. Of those output positions, 17,664 clipped at 255. Changing the weights to request rounded averaging passed a separate control. |
| [08. Two-input speed](results/08-two-input-throughput.json) | The two-input batch performed 33.6 million exact sums. Raw AVX2 took 35.8 ms versus 2,071 ms for NVENC plus NVDEC arithmetic. With encoded input, decode plus AVX2 took 1,129 ms. NVDEC arithmetic took 1,153 ms including preparation and 990 ms when prepacked. The prepared route returned 48 MiB of sums, while the CPU addition route received 96 MiB of input frames. |
| [09. GPU copy scheduling](results/09-copy-engine.json) | The native NVDEC interface used CUDA transfers into pinned host buffers. For prepared NVDEC arithmetic, alternating buffers reduced explicit copy wait from 29.4 ms to 1.1 ms and allowed decode submissions with copies pending. Total elapsed time remained about 264 ms. Including preparation, elapsed time went from 408 ms to 397 ms. |
| [10. Reused instructions and competing CPU work](results/10-instruction-reuse.json) | Reuse preserved the complete bitstream byte for byte and cut preparation from 134.5 ms to 13.3 ms. The cache started empty for each batch. Eight validated header signatures shared one instruction block. The concurrent test measured both the codec job and a separate CPU job on a two-vCPU VM. |
| [11. Completion-driven handoffs](results/11-completion-handoffs.json) | Two simultaneous NVENC → NVDEC → NVENC → NVDEC chains produced 256 exact frames in total. Both produced results while the source encoder was still open and before the second half of the input was released. Middle-stage logs confirm CUDA-resident video frames. Our harness wired the routes up front. FFmpeg and the driver handled codec submissions as outputs became ready. |
| [12. Dedicated optical flow on the T4](results/12-optical-flow.json) | Colab exposed `libnvidia-opticalflow.so.1` on driver 580.82.07. The NVIDIA Optical Flow API returned the exact median motion vector for all four image pairs. Across the four shifts, 91.7–100% of evaluated blocks landed within one pixel of the expected vector. |
| [13. Security engine XOR adapter, software proof](results/13-security-engine-xor-software.json) | AES-CBC decryption reproduced all 12,288 exact 16-byte XOR results in three cases. Two cases reused the same prepared left operands. This run used OpenSSL on the CPU. Jetson hardware execution is pending. |
| [14. NVDEC-to-NVENC XOR on the T4](results/14-nvdec-nvenc-xor-t4.json) | `h264_cuvid` decoded CUDA-resident 256 × 256 surfaces and `h264_nvenc` encoded the computed bit-planes. The CPU verification decoder confirmed all 24 output planes exactly: 1,572,864 XOR bit results across random, zero, and all-one second operands. The 1,574 ms preparation measurement included CPU construction of the encoded inputs **and** the CPU XOR oracle/expected planes. The decode-to-encode FFmpeg path took 844 ms. The corrected timing and usable byte output are in result 15. |
| [15. NVDEC-to-NVENC XOR byte output on the T4](results/15-nvdec-nvenc-xor-byte-output-t4.json) | The same T4 route reconstructed three exact 65,536-byte XOR arrays from 24 NVENC bit-planes. The separated timing was 1,519 ms CPU preparation, 830 ms GPU decode-to-encode path, 169 ms CPU output decode, and 238 ms byte reconstruction; raw inputs to usable bytes took 2,756 ms. Direct Python XOR of the same three arrays had an 11.7 ms median over seven runs on that Colab CPU. This is one hardware route run and does not establish a speedup, security-engine participation, or behavior alongside a busy CUDA workload. |

## Latest side-by-side result

The table reports medians of eight runs with native NVDEC and alternating pinned copies. The additional process completed repeatable PBKDF2-HMAC-SHA256 jobs. Its throughput is recorded during the codec run.

| Encoded-input path | Idle wall time | Idle host CPU time | Wall time with a CPU job | Other CPU jobs/s |
|---|---:|---:|---:|---:|
| Decode then AVX2 | 273.0 ms | 137.0 ms | 277.9 ms | 178.7 |
| NVDEC with original preparation | 398.9 ms | 249.7 ms | 664.0 ms | 139.4 |
| NVDEC with instruction reuse | 276.8 ms | 126.7 ms | 420.0 ms | 141.3 |
| NVDEC with the stream prepared ahead | 263.3 ms | 109.8 ms | 265.7 ms | 143.3 |

The concurrent runs had substantial variation. The reusable path came close to the CPU route on an otherwise idle VM. With the competing CPU process, the ordinary decoder plus AVX2 completed the batch sooner and the other process completed more jobs per second. The concurrent hardware experiment with an existing CUDA workload is pending.

## Optical flow accuracy

The [native probe](benchmarks/optical_flow_known_shift.cpp) used pairs of synthetic 256 × 256 grayscale images on the T4. Each second image was a shifted copy. The hardware API returned one vector per 4 × 4 block. The check excluded a 40-pixel border and disabled temporal hints so that each pair was evaluated independently. Passing required both median components to be within 0.5 pixels of the expected shift and at least 90% of evaluated vectors to be within one pixel on both axes.

| Expected shift (x, y) | Median (x, y) | Exact blocks | Blocks within one pixel |
|---|---|---:|---:|
| (0, 0) | (0, 0) | 1,936 / 1,936 | 1,936 / 1,936 |
| (8, −4) | (8, −4) | 1,933 / 1,936 | 1,933 / 1,936 |
| (−12, 8) | (−12, 8) | 1,936 / 1,936 | 1,936 / 1,936 |
| (4, 12) | (4, 12) | 1,760 / 1,936 | 1,775 / 1,936 |

The harness used CUDA driver calls for context, allocation, and transfers, and `nvOFExecute` for motion estimation. The test measured access and accuracy. Execution time and throughput with another CUDA workload are unmeasured.

## Security engine arithmetic

The [XOR adapter](benchmarks/security_engine_xor.py) prepares each 16-byte left operand `A` as `AES_encrypt(A)` with a fixed public test key. It arranges each pair as `[B, AES_encrypt(A)]` in the CBC input. CBC decryption of the second block decrypts it to `A` and XORs that value with the preceding block `B`. The adapter extracts one XOR result per pair, and the other output block is unused. Prepared left operands can be reused with new right operands.

The reference is direct bytewise CPU XOR. The software run verified 4,096 random pairs, then reused the prepared left side with all-zero and all-one right operands. All 12,288 results matched. This demonstrates bitwise XOR, equivalent to addition without carries, on synthetic operands. Each result needs two CBC input blocks and an AES preparation pass for new left operands.

The Jetson mode requests `ecb-aes-tegra` for preparation and `cbc-aes-tegra` for evaluation by their explicit driver names. It requires an existing Jetson Linux configuration that exposes those drivers through AF_ALG. Newer upstream Linux restricts AF_ALG to software implementations. The workspace attempt exited with status 2 and reported `[Errno 1] Operation not permitted` at the socket operation. Jetson hardware execution, physical engine activity, and timing remain unverified.

## How to read the results

- PNG reconstruction implements addition **modulo 256**. The H.264 operations use **saturating 8-bit addition**, clipping to 0–255. They are separate experiments.
- Each GPU result verifies a specified operation. The tests use structured input frames, and reference fidelity is checked for the fixture. The timing comparisons start from explicitly stated inputs. Preparation, transfers, and process startup are included where the record says they apply.
- Prepared-stream results exclude construction of the prediction instructions. Keep that distinction when comparing throughput.
- The native decoder harness and FFmpeg runs have different overheads. Compare methods within one experiment. Pending-copy counts and successful pipeline handoffs show submission order and readiness. Exact intervals of physical engine activity remain unmeasured.
- CUDA frame and copy APIs were used for memory and transfers. The weighted arithmetic was performed during NVDEC reconstruction. Codec internals are yet to be profiled for CUDA compute activity. Energy and board-level utilization are unmeasured.
- The benchmark started two fixed pipeline routes. Prioritization or selection between competing ready jobs remains to be tested.

## Reproduce the experiments

From the repository root, change to `experiments/codec-compute` before using the commands below. The original quick probes are in [run.py](run.py) and [colab_nvdec.ipynb](colab_nvdec.ipynb). Benchmarks import companion modules from the same directory.

Use Linux with Python 3.10+, FFmpeg, and, for the AVX2 comparisons, an x86-64 compiler with OpenMP. Hardware runs require NVIDIA drivers and FFmpeg with `h264_nvenc` and `h264_cuvid`. The native copy and reuse tests also require CUDA headers, a C++17 compiler, and NVIDIA driver libraries. They retrieve header files from the public NVIDIA SDK sample repository and record their SHA-256 hashes.

| Experiment | Command from this directory |
|---|---|
| Initial software experiments | `python3 benchmarks/initial_software_probe.py --output initial-software.json` |
| Offset arithmetic | `python3 benchmarks/weighted_codec_arithmetic.py --mode gpu --output offset.json` |
| Parallel feedback | `python3 benchmarks/codec_feedback_pipeline.py --mode gpu --chains 4 --hops 3 --output feedback.json` |
| Batched offsets | `python3 benchmarks/codec_batch_benchmark.py --hardware --frames 512 --repeats 3 --output batch.json` |
| One-frame offsets | `python3 benchmarks/codec_one_frame_benchmark.py --hardware --frames 512 --repeats 8 --output one-frame.json` |
| Two-input correctness | `python3 benchmarks/codec_two_input_add.py --hardware --output two-input.json` |
| Two-input throughput | `python3 benchmarks/codec_two_input_speed.py --hardware --pairs 512 --repeats 6 --output throughput.json` |
| Copy scheduling | `python3 benchmarks/codec_copy_benchmark.py --pairs 512 --repeats 6 --output copy.json` |
| Instruction reuse and concurrent CPU work | `python3 benchmarks/codec_reuse_concurrency.py --hardware --pairs 512 --repeats 8 --output reuse.json` |
| Completion-driven handoffs | `python3 benchmarks/codec_completion_chain.py --hardware --frames 128 --output handoffs.json` |
| Optical flow availability and accuracy | `python3 benchmarks/optical_flow_known_shift.py --output optical-flow.json` |
| Security engine arithmetic, software proof | `python3 benchmarks/security_engine_xor.py --backend openssl --output security-xor.json` |
| Security engine arithmetic, Jetson probe | `python3 benchmarks/security_engine_xor.py --backend jetson --output security-xor-jetson.json` |
| XOR bit-plane computation through NVDEC then NVENC | `python3 benchmarks/codec_xor_decode_encode.py --hardware --output xor-t4.json` |

For the initial software script, also install NumPy, Pillow, and Numba in the test environment. Its output path is configurable here. The probe captures the NVENC error if the hardware is unavailable. The other software-only paths use FFmpeg software codecs when invoked without `--hardware` or with `--mode cpu`, as their command-line help specifies. Each archived JSON distinguishes the available software and hardware records.

The optical flow script requires Linux, `nvidia-smi`, a C++17 compiler, CUDA headers under `/usr/local/cuda/include`, and a driver exposing `libcuda.so.1` and `libnvidia-opticalflow.so.1`. It downloads two public SDK headers pinned to a commit and records their SHA-256 hashes.

Use Python 3.10+ and the `openssl` executable for the security engine software proof. The Jetson mode uses Python's AF_ALG socket API and reports the first backend error on failure.

## Pending tests, in original order

2. **Profile CUDA compute use in the codec path.** Test NVENC, NVDEC and copy operations with features such as look-ahead and adaptive quantization disabled. Measure any programmable-core activity attributable to the codec path.
3. **Measure available media-engine capacity while CUDA is busy.** Compare an existing CUDA job alone and alongside codec work. Measure both throughputs in the same window, keeping the codec work on dedicated engines.
4. **Schedule several queued jobs.** Use completion signals to advance whichever job is ready when multiple jobs compete for an engine. Measure dispatch overhead and verify ordering and results under backpressure.
5. **Test exposed security hardware when available.** On Jetson Orin, verify the AES-CBC XOR adapter and compare preparation, reuse, and host overhead with direct CPU XOR. AES-GCM and SHA-256 are separate options for stream protection. The T4 SEC2 research identified firmware duties. Application-accessible crypto on that engine was not verified.

## Technical references

- [W3C PNG specification](https://www.w3.org/TR/png-3/) and [ITU-T H.264](https://www.itu.int/rec/T-REC-H.264/en)
- [NVIDIA NVDEC API guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html) and [NVENC API guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html)
- [NVIDIA CUDA guidance for overlapping transfers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) and [Optical Flow accelerator capabilities](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-application-note/index.html)
- [Jetson Orin security engine capabilities](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SO/JetsonOrinSeries.html#security-engine)
- [NIST SP 800-38A §6.2](https://doi.org/10.6028/NIST.SP.800-38A), [NVIDIA Tegra Security Engine driver notes](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/Security/DiskEncryption.html#crypto-engine-performance-considerations), and [Linux AF_ALG documentation](https://docs.kernel.org/crypto/userspace-if.html)
