#!/usr/bin/env python3
"""Measure three exact offset passes in a batched H.264 stream versus native CPU code."""

import argparse
import ctypes
import functools
import hashlib
import json
import os
import re
import resource
import statistics
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

from codec_feedback_pipeline import PLANS, seed
from weighted_codec_arithmetic import FFMPEG, HEIGHT, WIDTH, nal, reference_parameters, se, ue


HOPS = 3
BANDS = 8
LUMA = WIDTH * HEIGHT
FRAME = LUMA * 3 // 2
AUD = re.compile(rb"(?=\x00\x00\x00\x01\x09)|(?<!\x00)(?=\x00\x00\x01\x09)")
NATIVE = r"""
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

void add(const uint8_t *src, uint8_t *out, const int8_t *plans, int count, int threads) {
    enum { LUMA=65536, FRAME=98304, BAND=8192, BANDS=8, HOPS=3 };
    #pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
    for (int f=0; f<count; ++f) {
        const uint8_t *input = src + (size_t)f*FRAME;
        uint8_t *output = out + (size_t)f*FRAME;
        const int8_t *plan = plans + (size_t)f*BANDS*HOPS;
        for (int band=0; band<BANDS; ++band) {
            int a=plan[band], b=plan[BANDS+band], c=plan[2*BANDS+band];
            int start=band*BAND, end=start+BAND;
#if defined(__AVX2__)
            __m256i ap = _mm256_set1_epi8((char)(a>0?a:0));
            __m256i an = _mm256_set1_epi8((char)(a<0?-a:0));
            __m256i bp = _mm256_set1_epi8((char)(b>0?b:0));
            __m256i bn = _mm256_set1_epi8((char)(b<0?-b:0));
            __m256i cp = _mm256_set1_epi8((char)(c>0?c:0));
            __m256i cn = _mm256_set1_epi8((char)(c<0?-c:0));
            for (int i=start; i<end; i+=32) {
                __m256i v = _mm256_loadu_si256((const __m256i *)(input+i));
                v = _mm256_subs_epu8(_mm256_adds_epu8(v,ap),an);
                v = _mm256_subs_epu8(_mm256_adds_epu8(v,bp),bn);
                v = _mm256_subs_epu8(_mm256_adds_epu8(v,cp),cn);
                _mm256_storeu_si256((__m256i *)(output+i),v);
            }
#else
            for (int i=start; i<end; ++i) {
                int v=input[i]+a;
                v=(v<0?0:v>255?255:v)+b;
                v=(v<0?0:v>255?255:v)+c;
                output[i]=(uint8_t)(v<0?0:v>255?255:v);
            }
#endif
        }
        memcpy(output+LUMA,input+LUMA,FRAME-LUMA);
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


def run(command, data=None):
    result = subprocess.run(command, input=data, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}: {' '.join(map(str, command))}\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def encode_batch(raw, hardware):
    codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp", "-qp", "1",
             "-zerolatency", "1", "-forced-idr", "1", "-aud", "1"] if hardware else [
        "-c:v", "libx264", "-qp", "1", "-x264-params", "aud=1:repeat-headers=1:weightp=1:ref=1",
    ]
    return run(FFMPEG + [
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{WIDTH}x{HEIGHT}", "-r", "30", "-i", "pipe:0",
        "-profile:v", "main", "-bf", "0", "-g", "30", "-refs", "1",
        "-force_key_frames", "expr:1", *codec, "-f", "h264", "pipe:1",
    ], raw)


@functools.lru_cache(maxsize=64)
def make_instructions(parameters, schedule):
    sps, pps, frame_bits, poc_type, poc_bits = parameters
    new_pps = ue(pps)+ue(sps)+"00"+ue(0)*3+"100"+se(0)*3+"100"
    output = [nal(0x68, new_pps)]
    mb_count = WIDTH * HEIGHT // (16 * 16 * BANDS)
    for frame, offsets in enumerate(schedule, 1):
        output.append(nal(0x09, "001"))  # AUD, I/P slice types.
        for row, offset in enumerate(offsets):
            bits = ue(row*mb_count)+ue(0)+ue(pps)+f"{frame:0{frame_bits}b}"
            if poc_type == 0:
                bits += f"{2*frame:0{poc_bits}b}"
            bits += "00"+ue(0)*2+"1"+se(1)+se(offset)+"0"+"0"+se(0)+ue(1)+ue(mb_count)
            output.append(nal(0x41, bits))
    return b"".join(output)


def program(encoded, schedules):
    anchors = [part for part in AUD.split(encoded) if part]
    if len(anchors) != len(schedules):
        raise ValueError(f"Expected {len(schedules)} IDR access units, got {len(anchors)}")
    # Every IDR is self-contained. Its dependent P frames stay in the decoder.
    return b"".join(
        anchor+make_instructions(reference_parameters(anchor), schedule)
        for anchor, schedule in zip(anchors, schedules, strict=True)
    )


def decode_batch(encoded, hardware, select_final=False):
    inputs = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid"] if hardware else ["-c:v", "h264"]
    filters = [r"select=eq(mod(n\,4)\,3)"] if select_final else []
    if hardware:
        filters += ["hwdownload", "format=nv12", "format=yuv420p"]
    video_filter = ["-vf", ",".join(filters)] if filters else []
    return run(FFMPEG + inputs + ["-f", "h264", "-i", "pipe:0"] + video_filter + [
        "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-threads:v", "1", "-f", "rawvideo", "pipe:1",
    ], encoded)


def compile_cpu(directory):
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    source = directory / "baseline.c"
    target = directory / "baseline.so"
    source.write_text(NATIVE)
    run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", str(source), "-o", str(target)])
    library = ctypes.CDLL(str(target))
    library.add.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    library.add.restype = None
    return library


def cpu_apply(library, raw, plan, count, threads):
    if len(raw) != count * FRAME:
        raise ValueError(f"Expected {count*FRAME} raw bytes, got {len(raw)}")
    result = bytearray(len(raw))
    target = (ctypes.c_ubyte * len(result)).from_buffer(result)
    library.add(raw, target, plan, count, threads)
    return result


def host_cpu():
    scopes = (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    return sum(t.ru_utime + t.ru_stime for t in map(resource.getrusage, scopes))


def checked(test, output, reference):
    if output != reference:
        count = sum(a != b for a, b in zip(output, reference))
        raise AssertionError(f"{test}: {len(output)} vs {len(reference)} bytes and {count} differing overlapping bytes")


def benchmark(count, repeats, hardware):
    if (WIDTH, HEIGHT) != (256, 256):
        raise ValueError("Native baseline expects 256×256 input")
    threads = min(2, len(os.sched_getaffinity(0)))
    schedules = [tuple(PLANS[(frame+hop) % len(PLANS)] for hop in range(HOPS)) for frame in range(count)]
    plan = bytes(offset & 255 for schedule in schedules for band_offsets in schedule for offset in band_offsets)
    raw = b"".join(seed(frame) for frame in range(count))
    with tempfile.TemporaryDirectory() as path:
        lib = compile_cpu(Path(path))
        expected = cpu_apply(lib, raw, plan, count, 1)
        # Verify the native implementation independently for the first frame.
        scalar = raw[:FRAME]
        for band_offsets in schedules[0]:
            bands = [scalar[i*8192:(i+1)*8192] for i in range(BANDS)]
            scalar = b"".join(block.translate(bytes(min(255,max(0,x+offset)) for x in range(256)))
                              for block, offset in zip(bands, band_offsets, strict=True)) + scalar[LUMA:]
        checked("native CPU vs scalar arithmetic", expected[:FRAME], scalar)
        anchors = encode_batch(raw, hardware)
        checked("encode/decode input fidelity", decode_batch(anchors, hardware), raw)
        cases = {
            "raw_cpu_1_thread": lambda: cpu_apply(lib, raw, plan, count, 1),
            "raw_cpu_up_to_2_threads": lambda: cpu_apply(lib, raw, plan, count, threads),
            "raw_codec_encode_and_compute": lambda: decode_batch(program(encode_batch(raw, hardware), schedules), hardware, True),
            "encoded_decode_then_cpu": lambda: cpu_apply(lib, decode_batch(anchors, hardware), plan, count, threads),
            "encoded_decode_and_compute": lambda: decode_batch(program(anchors, schedules), hardware, True),
        }
        # Warm each path with the same inputs, then reverse alternate timing orders.
        for name, evaluate in cases.items():
            checked(name, evaluate(), expected)
        scores = {name: [] for name in cases}
        for repetition in range(repeats):
            order = list(cases) if repetition % 2 == 0 else list(reversed(cases))
            for name in order:
                cpu, tick = host_cpu(), time.perf_counter()
                result = cases[name]()
                walls, cpus = time.perf_counter()-tick, host_cpu()-cpu
                checked(name, result, expected)
                scores[name].append({"wall_ms":round(walls*1000,4), "host_cpu_ms":round(cpus*1000,4)})
        for name, entries in scores.items():
            scores[name] = {"runs":entries, **{f"median_{key}":round(statistics.median(r[key] for r in entries),4)
                                             for key in ("wall_ms","host_cpu_ms")}}
        return {
            "hardware": hardware, "frames": count, "passes": HOPS, "luma_operations": count*LUMA*HOPS,
            "native_avx2": bool(lib.avx2()), "cpu_threads_parallel": threads,
            "encoded_bytes": len(anchors), "output_bytes": len(expected), "outputs_exact": True,
            "sha256": hashlib.sha256(expected).hexdigest(), "timings": scores,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark(args.frames, args.repeats, args.hardware)
    result["cpu_model"] = next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None)
    if args.hardware:
        result["gpu"] = run(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"]).decode().strip()
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("RESULT="+json.dumps(result,separators=(",",":")))


if __name__ == "__main__":
    main()
