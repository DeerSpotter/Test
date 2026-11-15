#!/usr/bin/env python3
"""Reuse verified H.264 instructions and measure concurrent CPU work."""

import argparse
import ctypes as C
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from codec_copy_benchmark import Stats, load_backend, oracle
from codec_two_input_add import read_vcl_header, two_reference_stream, units
from codec_two_input_speed import AUD, LENGTH, LUMA, checked, encode_batch, fixture, host_cpu, prepare_pairs, run
from weighted_codec_arithmetic import reference_parameters


def relevant_headers(frame):
    headers = []
    for unit in units(frame):
        kind = unit[0] & 31
        if kind in (7, 8):
            headers.append(unit)
        elif kind in (1, 5):
            # The cache admits a prefix only after fully reading its relevant fields.
            headers.append(unit[:32])
    return tuple(headers)


def prepare_reusing(anchors, pairs):
    frames = [frame for frame in AUD.split(anchors) if frame]
    if len(frames) != 2*pairs:
        raise ValueError(f"Expected {2*pairs} coded access units, got {len(frames)}")
    cache = {}
    output = []
    misses = 0
    for first, second in zip(frames[::2], frames[1::2], strict=True):
        left, right = relevant_headers(first), relevant_headers(second)
        key = left, right
        tail = cache.get(key)
        if tail is None:
            misses += 1
            pair = first + second
            stream, _ = two_reference_stream(pair, 2, 2)
            tail = stream[len(pair):]
            _, _, frame_bits, poc_type, poc_bits = reference_parameters(first)
            try:
                for headers in (left, right):
                    vcl = next(unit for unit in headers if (unit[0] & 31) in (1, 5))
                    read_vcl_header(vcl, frame_bits, poc_type, poc_bits)
            except (ValueError, StopIteration):
                # Keep the checked result but do not reuse an incomplete prefix.
                pass
            else:
                cache[key] = tail
        output.extend((first, second, tail))
    return b"".join(output), {
        "pair_count": pairs, "instructions_generated": misses,
        "reused_pairs": pairs-misses, "validated_signatures": len(cache),
        "distinct_instruction_blocks": len(set(cache.values())),
    }


def compare_packers(anchors, pairs, repeats):
    cold = []
    reused = []
    for _ in range(repeats):
        t = perf_counter()
        original = prepare_pairs(anchors, pairs)
        cold.append((perf_counter()-t)*1000)
        t = perf_counter()
        optimized, info = prepare_reusing(anchors, pairs)
        reused.append((perf_counter()-t)*1000)
        checked("complete reusable H.264 stream", optimized, original)
    return optimized, {**info, "byte_identical": True, "stream_bytes": len(optimized),
        "original_ms": cold, "reuse_ms": reused,
        "median_original_ms": median(cold), "median_reuse_ms": median(reused)}


def hash_worker(ready, stop, count):
    ready.set()
    while not stop.is_set():
        # A repeatable independent CPU job, with no GPU or memory allocation in its loop.
        hashlib.pbkdf2_hmac("sha256", b"independent CPU job", b"codec experiment", 6000)
        count.value += 1


def measure(lib, anchors, programmed, expected, pairs, repeats):
    threads = min(2, len(os.sched_getaffinity(0)))
    result_area = C.create_string_buffer(pairs * LENGTH)
    error = C.create_string_buffer(1024)
    runner = lib.make_runner(pairs, error, len(error))
    if not runner:
        raise RuntimeError(error.value.decode())
    try:
        slots = [lib.output_slot(runner, i) for i in range(2)]

        def evaluate(name, counter):
            start_count = counter.value if counter is not None else 0
            cpu, start = host_cpu(), perf_counter()
            sums = name != "decode_then_avx2"
            if name == "nvdec_rebuild":
                encoded = prepare_pairs(anchors, pairs)
            elif name == "nvdec_reuse":
                encoded, _ = prepare_reusing(anchors, pairs)
            else:
                encoded = programmed if sums else anchors
            stats = Stats()
            code = lib.run_decoder(runner, encoded, len(encoded), 1, sums,
                                   C.byref(stats), error, len(error))
            if code:
                raise RuntimeError(error.value.decode())
            if not sums:
                lib.add_on_cpu(runner, pairs, result_area, threads)
            elapsed, active_cpu = perf_counter()-start, host_cpu()-cpu
            jobs = (counter.value-start_count) if counter is not None else None
            assert stats.copied == pairs*(1 if sums else 2)
            assert stats.displayed == pairs*(3 if sums else 2)
            result = (b"".join(C.string_at(slots[i%2]+(i//2)*LENGTH,LENGTH) for i in range(pairs))
                      if sums else result_area.raw)
            checked(name + " decoded sums", result, expected)
            return {
                "wall_ms":round(1000*elapsed,4), "host_cpu_ms":round(1000*active_cpu,4),
                "background_hashes":jobs,
                "background_hashes_per_s":round(jobs/elapsed,4) if jobs is not None else None,
            }

        names = ["decode_then_avx2","nvdec_rebuild","nvdec_reuse","nvdec_prebuilt"]
        for name in names:
            evaluate(name, None)
        output = {}
        for load in ("idle", "one_cpu_worker"):
            context = mp.get_context("spawn")
            ready, stop = context.Event(), context.Event()
            count = context.RawValue("q", 0)
            worker = None
            if load == "one_cpu_worker":
                worker = context.Process(target=hash_worker,args=(ready,stop,count))
                worker.start()
            try:
                if worker:
                    if not ready.wait(10):
                        raise RuntimeError("The independent CPU process failed to start")
                data = {name: [] for name in names}
                for r in range(repeats):
                    first = r % len(names)
                    for name in names[first:] + names[:first]:
                        data[name].append(evaluate(name, count if worker else None))
                output[load] = {name: {"runs": rows, **{
                    "median_"+key: median(r[key] for r in rows)
                    for key in ("wall_ms","host_cpu_ms","background_hashes_per_s")
                    if rows[0][key] is not None
                }} for name,rows in data.items()}
            finally:
                if worker:
                    stop.set()
                    worker.join(10)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
                        raise RuntimeError("The background process did not stop")
                    if worker.exitcode != 0:
                        raise RuntimeError(f"Background process exit code {worker.exitcode}")
        return output
    finally:
        lib.destroy_runner(runner)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    if args.pairs < 1 or args.repeats < 1:
        parser.error("pairs and repeats must be positive")
    anchors=encode_batch(fixture(args.pairs), args.hardware)
    prepared, prep=compare_packers(anchors,args.pairs,args.repeats)
    expected=oracle(args.pairs)
    output={"pairs":args.pairs,"sums":args.pairs*LUMA,"prep":prep,
            "reference_sha256":hashlib.sha256(expected).hexdigest(),"hardware":args.hardware}
    if args.hardware:
        output["gpu"]=run(["nvidia-smi","--query-gpu=name,driver_version",
                           "--format=csv,noheader"]).decode().strip()
        output["available_logical_cpus"]=len(os.sched_getaffinity(0))
        with TemporaryDirectory() as temp:
            lib, hashes=load_backend(Path(temp))
            output["tests"]=measure(lib,anchors,prepared,expected,args.pairs,args.repeats)
            output["nvidia_header_sha256"]=hashes
        output["all_decoded_bytes_exact"]=True
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    summary=dict(output)
    if args.hardware:
        summary["tests"]={load:{name:{key:value for key,value in stats.items() if key!="runs"}
                                for name,stats in cases.items()} for load,cases in output["tests"].items()}
    print("REUSE_SUMMARY="+json.dumps(summary,separators=(",",":")))
    if args.hardware:
        for load,cases in output["tests"].items():
            for name,stats in cases.items():
                print("REUSE_TRIALS="+json.dumps([load,name,[list(row.values()) for row in stats["runs"]]],separators=(",",":")))


if __name__=="__main__":
    main()
