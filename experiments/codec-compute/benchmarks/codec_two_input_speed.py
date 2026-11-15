#!/usr/bin/env python3
"""Benchmark batched two-operand saturating addition on a CPU and NVDEC."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from codec_batch_benchmark import AUD, checked, host_cpu, run
from codec_two_input_add import LENGTH, LUMA, operands, two_reference_stream
from weighted_codec_arithmetic import FFMPEG, HEIGHT, WIDTH


NATIVE = r"""
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

void add_pairs(const uint8_t *input, uint8_t *output, int count, int threads) {
    enum { LUMA=65536, FRAME=98304 };
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int i=0; i<count; i++) {
        const uint8_t *a=input+(size_t)i*2*FRAME, *b=a+FRAME;
        uint8_t *dst=output+(size_t)i*FRAME;
#if defined(__AVX2__)
        for (int j=0; j<LUMA; j+=32) {
            __m256i x=_mm256_loadu_si256((const __m256i *)(a+j));
            __m256i y=_mm256_loadu_si256((const __m256i *)(b+j));
            _mm256_storeu_si256((__m256i *)(dst+j),_mm256_adds_epu8(x,y));
        }
#else
        for (int j=0; j<LUMA; j++) {
            unsigned int sum=(unsigned int)a[j]+b[j];
            dst[j]=(uint8_t)(sum>255?255:sum);
        }
#endif
        memcpy(dst+LUMA,a+LUMA,FRAME-LUMA);
    }
}
int avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}
"""


def fixture(pairs):
    first, second = operands()
    row, a_chroma = first[:WIDTH], first[LUMA:]
    rows, b_chroma = second[:LUMA], second[LUMA:]
    variants = []
    for phase in range(8):
        x = phase * 16
        y = ((phase * 3) % 16) * 16 * WIDTH
        a = (row[x:] + row[:x]) * HEIGHT + a_chroma
        b = rows[y:] + rows[:y] + b_chroma
        variants.append(a + b)
    return b"".join(variants[i % len(variants)] for i in range(pairs))


def encode_batch(raw, hardware):
    if hardware:
        options = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp",
                   "-qp", "1", "-aud", "1", "-forced-idr", "1"]
    else:
        options = ["-c:v", "libx264", "-qp", "1", "-x264-params",
                   "aud=1:repeat-headers=1:b-adapt=0:scenecut=0:open-gop=0"]
    return run(FFMPEG + [
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", "30", "-i", "pipe:0", "-profile:v", "main", "-g", "30",
        "-bf", "1", "-refs", "2", "-force_key_frames", "expr:eq(mod(n,2),0)",
        *options, "-f", "h264", "pipe:1",
    ], raw)


def prepare_pairs(anchors, pairs):
    frames = [unit for unit in AUD.split(anchors) if unit]
    if len(frames) != 2*pairs:
        raise AssertionError(f"Expected two coded access units per pair, got {len(frames)} for {pairs}")
    output = []
    for index in range(0, len(frames), 2):
        try:
            stream, _ = two_reference_stream(frames[index]+frames[index+1], 2, 2)
        except ValueError as error:
            raise ValueError(f"Input pair {index//2}: {error}") from error
        output.append(stream)
    return b"".join(output)


def decode_batch(bitstream, hardware, sums_only=False):
    decoder = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
               "-c:v", "h264_cuvid"] if hardware else ["-c:v", "h264"]
    filters = [r"select=eq(mod(n\,3)\,1)"] if sums_only else []
    if hardware:
        filters += ["hwdownload", "format=nv12", "format=yuv420p"]
    return run(FFMPEG + decoder + ["-f", "h264", "-i", "pipe:0"] +
               (["-vf", ",".join(filters)] if filters else []) +
               ["-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
                "-threads:v", "1", "-f", "rawvideo", "pipe:1"], bitstream)


def load_cpu(folder):
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    source, target = folder / "add_pairs.c", folder / "add_pairs.so"
    source.write_text(NATIVE)
    run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
         str(source), "-o", str(target)])
    library = ctypes.CDLL(str(target))
    library.add_pairs.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.add_pairs.restype = None
    return library


def benchmark(pairs, repeats, hardware):
    if (WIDTH, HEIGHT, LENGTH) != (256, 256, 98304):
        raise ValueError("The AVX2 baseline expects 256×256 YUV420p inputs")
    raw = fixture(pairs)
    anchors = encode_batch(raw, hardware)
    programmed = prepare_pairs(anchors, pairs)
    checked("Exact recovery of encoded operands", decode_batch(anchors, hardware), raw)
    threads = min(2, len(os.sched_getaffinity(0)))
    with TemporaryDirectory() as directory:
        lib = load_cpu(Path(directory))

        def add(source, worker_count=threads):
            if len(source) != len(raw):
                raise AssertionError(f"Expected {len(raw)} operand bytes, got {len(source)}")
            result = bytearray(pairs * LENGTH)
            target = (ctypes.c_ubyte * len(result)).from_buffer(result)
            lib.add_pairs(source, target, pairs, worker_count)
            return result

        reference = add(raw)
        a, b = raw[:LENGTH], raw[LENGTH:2*LENGTH]
        scalar = bytes(min(255,x+y) for x,y in zip(a[:LUMA], b[:LUMA], strict=True)) + a[LUMA:]
        checked("AVX2 vs scalar for the first pair", reference[:LENGTH], scalar)
        cases = {
            "raw_avx2_1_thread": lambda: add(raw, 1),
            "raw_avx2_2_threads": lambda: add(raw),
            "raw_nvenc_then_nvdec_sum": lambda: decode_batch(prepare_pairs(encode_batch(raw, hardware), pairs), hardware, True),
            "encoded_nvdec_then_avx2": lambda: add(decode_batch(anchors, hardware)),
            "encoded_nvdec_sum_with_packing": lambda: decode_batch(prepare_pairs(anchors, pairs), hardware, True),
            "encoded_nvdec_sum_prepacked": lambda: decode_batch(programmed, hardware, True),
        }
        for label, execute in cases.items():
            checked(label, execute(), reference)

        names = list(cases)
        timings = {name: [] for name in names}
        for i in range(repeats):
            first = i % len(names)
            for name in names[first:] + names[:first]:
                cpu_start, start = host_cpu(), perf_counter()
                result = cases[name]()
                elapsed, cpu_seconds = perf_counter()-start, host_cpu()-cpu_start
                checked(name, result, reference)
                timings[name].append({"wall_ms":round(1000*elapsed,4),
                                      "host_cpu_ms":round(1000*cpu_seconds,4)})
        avx2 = bool(lib.avx2())
    return {
        "hardware": hardware, "pairs": pairs, "luma_sums": pairs*LUMA, "avx2": avx2,
        "max_cpu_threads": threads, "all_outputs_exact": True,
        "encoded_operands_exact": True, "output_sha256": hashlib.sha256(reference).hexdigest(),
        "raw_input_bytes": len(raw), "encoded_operand_bytes": len(anchors),
        "encoded_programmed_bytes": len(programmed), "output_bytes": len(reference),
        "timings": {name: {"runs": rows, **{
            f"median_{metric}": round(median(row[metric] for row in rows),4)
            for metric in ("wall_ms","host_cpu_ms")}} for name, rows in timings.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or args.repeats < 1:
        parser.error("pairs and repeats must be positive")
    result = benchmark(args.pairs, args.repeats, args.hardware)
    if args.hardware:
        result["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version",
                             "--format=csv,noheader"]).decode().strip()
    result["cpu"] = next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), None)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("TWO_INPUT_SPEED=" + json.dumps(result,separators=(",",":")))


if __name__ == "__main__":
    main()
