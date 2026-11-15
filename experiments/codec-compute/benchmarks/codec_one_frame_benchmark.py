#!/usr/bin/env python3
"""Test one weighted H.264 frame against one AVX2 saturating offset pass."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from codec_batch_benchmark import (
    FFMPEG, FRAME, LUMA, PLANS, checked, encode_batch, host_cpu, program, run, seed,
)


NATIVE = r"""
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

void offsets(const uint8_t *input, uint8_t *output,
             const int8_t *plans, int frames, int threads) {
    enum { FRAME=98304, LUMA=65536, BAND=8192, BANDS=8 };
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int frame=0; frame<frames; frame++) {
        const uint8_t *src=input+(size_t)frame*FRAME;
        uint8_t *dst=output+(size_t)frame*FRAME;
        for (int band=0; band<BANDS; band++) {
            int offset=plans[frame*BANDS+band], start=band*BAND, end=start+BAND;
#if defined(__AVX2__)
            __m256i amount=_mm256_set1_epi8((char)(offset<0?-offset:offset));
            for (int i=start; i<end; i+=32) {
                __m256i old=_mm256_loadu_si256((const __m256i *)(src+i));
                __m256i next=offset<0?_mm256_subs_epu8(old,amount):_mm256_adds_epu8(old,amount);
                _mm256_storeu_si256((__m256i *)(dst+i),next);
            }
#else
            for (int i=start; i<end; i++) {
                int value=src[i]+offset;
                dst[i]=(uint8_t)(value<0?0:value>255?255:value);
            }
#endif
        }
        memcpy(dst+LUMA,src+LUMA,FRAME-LUMA);
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


def cpu_library(folder):
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    source, target = folder / "one_pass.c", folder / "one_pass.so"
    source.write_text(NATIVE)
    run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
         str(source), "-o", str(target)])
    library = ctypes.CDLL(str(target))
    library.offsets.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
    library.offsets.restype = None
    return library


def decode(encoded, hardware, predicted=False):
    decoder = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid"] if hardware else ["-c:v", "h264"]
    filters = [r"select=eq(mod(n\,2)\,1)"] if predicted else []
    if hardware:
        filters += ["hwdownload", "format=nv12", "format=yuv420p"]
    return run(FFMPEG + decoder + ["-f", "h264", "-i", "pipe:0"] +
               (["-vf", ",".join(filters)] if filters else []) +
               ["-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-threads:v", "1",
                "-f", "rawvideo", "pipe:1"], encoded)


def summarize(rows):
    return {
        "runs": rows,
        "median_wall_ms": round(median(r["wall_ms"] for r in rows), 4),
        "median_host_cpu_ms": round(median(r["host_cpu_ms"] for r in rows), 4),
    }


def benchmark(frames, repeats, hardware):
    raw = b"".join(seed(i) for i in range(frames))
    schedules = [(PLANS[i % len(PLANS)],) for i in range(frames)]
    plan = bytes(offset & 255 for (offsets,) in schedules for offset in offsets)
    anchors = encode_batch(raw, hardware)
    prebuilt = program(anchors, schedules)
    threads = min(2, len(os.sched_getaffinity(0)))
    with TemporaryDirectory() as directory:
        lib = cpu_library(Path(directory))

        def apply_cpu(source):
            if len(source) != frames * FRAME:
                raise AssertionError(f"Expected {frames * FRAME} decoded bytes, got {len(source)}")
            result = bytearray(len(source))
            target = (ctypes.c_ubyte * len(result)).from_buffer(result)
            lib.offsets(source, target, plan, frames, threads)
            return result

        expected = apply_cpu(raw)
        band = LUMA // 8
        scalar_sample = bytes(min(255,max(0,s + PLANS[0][i // band]))
                              for i, s in enumerate(raw[:LUMA])) + raw[LUMA:FRAME]
        checked("AVX2 vs scalar first frame", expected[:FRAME], scalar_sample)
        methods = {
            "plain_decode": lambda: decode(anchors, hardware),
            "decode_then_avx2": lambda: apply_cpu(decode(anchors, hardware)),
            "weighted_nvdec_with_packing": lambda: decode(program(anchors, schedules), hardware, True),
            "weighted_nvdec_prepacked": lambda: decode(prebuilt, hardware, True),
        }
        for name, execute in methods.items():
            checked(name, execute(), raw if name == "plain_decode" else expected)

        scores = {name: [] for name in methods}
        names = list(methods)
        for index in range(repeats):
            first = index % len(names)
            for name in names[first:] + names[:first]:
                cpu, tick = host_cpu(), perf_counter()
                result = methods[name]()
                elapsed, consumed = perf_counter() - tick, host_cpu() - cpu
                checked(name, result, raw if name == "plain_decode" else expected)
                scores[name].append({"wall_ms": round(elapsed*1000, 4),
                                     "host_cpu_ms": round(consumed*1000, 4)})

    return {
        "hardware": hardware,
        "frames": frames,
        "luma_operations": frames * LUMA,
        "avx2": bool(lib.avx2()),
        "cpu_threads": threads,
        "outputs_exact": True,
        "output_sha256": hashlib.sha256(expected).hexdigest(),
        "anchor_bytes": len(anchors),
        "weighted_bytes": len(prebuilt),
        "timings": {name: summarize(scores[name]) for name in methods},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.repeats < 1:
        parser.error("frames and repeats must be positive")
    result = benchmark(args.frames, args.repeats, args.hardware)
    if args.hardware:
        result["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version",
                             "--format=csv,noheader"]).decode().strip()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("ONE_FRAME_RESULT=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
