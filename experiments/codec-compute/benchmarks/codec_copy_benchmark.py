#!/usr/bin/env python3
"""Measure pinned serial versus pipelined downloads through native NVDEC."""

import argparse
import ctypes as C
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from codec_two_input_speed import LENGTH, LUMA, checked, encode_batch, fixture, host_cpu, prepare_pairs, run


HEADERS = "https://raw.githubusercontent.com/NVIDIA/video-sdk-samples/master/Samples/NvCodec/NvDecoder/"
SOURCE = Path(__file__).with_name("codec_copy_engine.cpp")


class Stats(C.Structure):
    _fields_ = [
        ("displayed", C.c_int), ("copied", C.c_int),
        ("decode_submissions", C.c_int), ("submissions_with_copy_pending", C.c_int),
        ("map_ms", C.c_double), ("sync_wait_ms", C.c_double),
    ]


def load_backend(directory):
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    hashes = {}
    for name in ("nvcuvid.h", "cuviddec.h"):
        with urllib.request.urlopen(HEADERS + name, timeout=20) as response:
            data = response.read()
        if b"NVIDIA Corporation" not in data:
            raise ValueError(f"Unexpected NVIDIA header contents for {name}")
        (directory / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    libs = []
    for name in ("libcuda.so.1", "libnvcuvid.so.1"):
        C.CDLL(name)
        prefix = name.removesuffix(".1")
        mapped = (line.rsplit(maxsplit=1)[-1] for line in Path("/proc/self/maps").read_text().splitlines())
        paths = [path for path in mapped if path.startswith("/") and Path(path).name.startswith(prefix)]
        if not paths:
            raise RuntimeError(f"Could not locate the loaded driver library: {name}")
        libs.append(paths[0])
    so = directory / "codec_copy_engine.so"
    run(["g++", "-std=c++17", "-O3", "-mavx2", "-fopenmp", "-fPIC", "-shared",
         "-I", str(directory), "-I", "/usr/local/cuda/include",
         str(SOURCE), *libs, "-o", str(so)])
    lib = C.CDLL(str(so))
    for name, args, result in (
        ("make_runner", [C.c_int,C.c_char_p,C.c_size_t], C.c_void_p),
        ("run_decoder", [C.c_void_p,C.c_char_p,C.c_size_t,C.c_int,C.c_int,C.POINTER(Stats),
                         C.c_char_p,C.c_size_t], C.c_int),
        ("output_slot", [C.c_void_p,C.c_int], C.c_void_p),
        ("async_engines", [C.c_void_p], C.c_int),
        ("add_on_cpu", [C.c_void_p,C.c_int,C.c_void_p,C.c_int], None),
        ("destroy_runner", [C.c_void_p], None),
    ):
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib, hashes


def oracle(pairs):
    raw = fixture(min(pairs, 8))
    rows = []
    for start in range(0, len(raw), 2*LENGTH):
        a, b = raw[start:start+LENGTH], raw[start+LENGTH:start+2*LENGTH]
        y = bytes(min(255, x+y) for x, y in zip(a[:LUMA], b[:LUMA], strict=True))
        rows.append(y + a[LUMA:])
    return b"".join(rows[i % len(rows)] for i in range(pairs))


def benchmark(lib, pairs, repeats):
    initial = encode_batch(fixture(pairs), hardware=True)
    prepacked = prepare_pairs(initial, pairs)
    expected = oracle(pairs)
    threads = min(2, len(os.sched_getaffinity(0)))
    result_area = C.create_string_buffer(pairs * LENGTH)
    error = C.create_string_buffer(1024)
    runner = lib.make_runner(pairs, error, len(error))
    if not runner:
        raise RuntimeError(error.value.decode())
    try:
        slots = [lib.output_slot(runner, i) for i in range(2)]

        def perform(sums, async_copy, pack):
            cpu, start = host_cpu(), perf_counter()
            encoded = prepare_pairs(initial, pairs) if pack else prepacked if sums else initial
            stats = Stats()
            status = lib.run_decoder(runner, encoded, len(encoded), async_copy, sums,
                                     C.byref(stats), error, len(error))
            if status:
                raise RuntimeError(error.value.decode())
            if not sums:
                lib.add_on_cpu(runner, pairs, result_area, threads)
            elapsed, cpu_used = perf_counter()-start, host_cpu()-cpu
            if stats.copied != pairs * (1 if sums else 2):
                raise AssertionError(f"Unexpected copied frame count {stats.copied}")
            if stats.displayed != pairs * (3 if sums else 2):
                raise AssertionError(f"Unexpected presented frame count {stats.displayed}")
            result = (b"".join(C.string_at(slots[i%2]+(i//2)*LENGTH,LENGTH) for i in range(pairs))
                      if sums else result_area.raw)
            checked("decoded output", result, expected)
            return {
                "wall_ms": round(1000*elapsed,4), "host_cpu_ms": round(1000*cpu_used,4),
                "displayed_frames": stats.displayed, "returned_frames": stats.copied,
                "bytes_downloaded": stats.copied*LENGTH,
                "decode_submissions": stats.decode_submissions,
                "submissions_with_copy_pending": stats.submissions_with_copy_pending,
                "frame_mapping_ms": round(stats.map_ms,4),
                "copy_sync_wait_ms": round(stats.sync_wait_ms,4),
            }

        cases = {
            "cpu_math_blocking": (False,False,False),
            "cpu_math_alternating": (False,True,False),
            "nvdec_prepared_blocking": (True,False,False),
            "nvdec_prepared_alternating": (True,True,False),
            "nvdec_with_preparation_blocking": (True,False,True),
            "nvdec_with_preparation_alternating": (True,True,True),
        }
        timings = {name: [] for name in cases}
        for args in cases.values():
            perform(*args)
        names = list(cases)
        for i in range(repeats):
            first = i % len(names)
            for name in names[first:] + names[:first]:
                timings[name].append(perform(*cases[name]))
        measurements = {}
        for name, rows in timings.items():
            measurements[name] = {
                "runs": rows,
                **{f"median_{key}": round(median(row[key] for row in rows),4)
                   for key in ("wall_ms","host_cpu_ms","frame_mapping_ms","copy_sync_wait_ms")},
                "total_submissions_with_copy_pending":
                    sum(row["submissions_with_copy_pending"] for row in rows),
            }
        return {"pairs": pairs, "repeats": repeats, "luma_sums": pairs*LUMA,
                "avx2_threads": threads, "reported_cuda_async_engines": lib.async_engines(runner),
                "all_modes_exact": True, "sha256": hashlib.sha256(expected).hexdigest(),
                "measurements": measurements}
    finally:
        lib.destroy_runner(runner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or args.repeats < 1:
        parser.error("pairs and repeats must be positive")
    with TemporaryDirectory() as temporary:
        lib, hashes = load_backend(Path(temporary))
        result = benchmark(lib, args.pairs, args.repeats)
    result["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version",
                         "--format=csv,noheader"]).decode().strip()
    result["nvidia_header_sha256"] = hashes
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = dict(result)
    summary["measurements"] = {
        name: {key: val for key, val in metrics.items() if key != "runs"}
        for name, metrics in result["measurements"].items()
    }
    print("COPY_SUMMARY="+json.dumps(summary,separators=(",",":")))
    for name, case in result["measurements"].items():
        fields = list(case["runs"][0])
        print("COPY_TRIALS="+json.dumps([name,fields,[list(row.values()) for row in case["runs"]]],
                                       separators=(",",":")))


if __name__ == "__main__":
    main()
