#!/usr/bin/env python3
"""Compare serial and concurrent feedback loops through NVENC and NVDEC."""

import argparse
import base64
import ctypes
import hashlib
import json
import resource
import statistics
import threading
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory

from weighted_codec_arithmetic import HEIGHT, LEVELS, WIDTH, decode, encode, prediction_stream, run


PLANS = (
    (0, 19, -37, 63, -128, 127, -1, 1),
    (1, -1, 7, -17, 31, -41, 53, -67),
    (-16, 32, 0, -24, 8, -8, 16, 40),
    (9, 33, -21, 45, -57, 0, 3, -3),
)
LUMA = WIDTH * HEIGHT


def seed(chain):
    levels = LEVELS[chain % len(LEVELS):] + LEVELS[:chain % len(LEVELS)]
    row = bytes(level for level in levels for _ in range(WIDTH // len(LEVELS)))
    return row * HEIGHT + bytes([128]) * (LUMA // 2)


def expected_result(reference, plan):
    slice_bytes = LUMA // len(plan)
    result = []
    for index, offset in enumerate(plan):
        samples = reference[index * slice_bytes:(index + 1) * slice_bytes]
        table = bytes(max(0, min(255, n + offset)) for n in range(256))
        result.append(samples.translate(table))
    return b"".join(result) + reference[LUMA:]


class VideoTelemetry:
    """Device-wide NVML windows, independent of the subprocess timings."""

    def __init__(self, hardware, origin):
        self.samples = []
        self.error = None
        self.origin = origin
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect) if hardware else None

    def __enter__(self):
        if self.thread:
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join()

    def collect(self):
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            device = ctypes.c_void_p()
            check = self.check
            check(lib.nvmlInit_v2(), "nvmlInit_v2")
            try:
                check(lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device)), "device 0")
                while not self.stop.is_set():
                    sample = [round((time.perf_counter() - self.origin) * 1000, 1)]
                    for engine in ("Encoder", "Decoder"):
                        percent, period = ctypes.c_uint(), ctypes.c_uint()
                        query = getattr(lib, f"nvmlDeviceGet{engine}Utilization")
                        check(query(device, ctypes.byref(percent), ctypes.byref(period)), engine)
                        sample.extend((percent.value, period.value))
                    self.samples.append(sample)
                    self.stop.wait(0.05)
            finally:
                lib.nvmlShutdown()
        except (OSError, AttributeError, RuntimeError) as error:
            self.error = str(error)

    @staticmethod
    def check(status, operation):
        if status:
            raise RuntimeError(f"NVML {operation} returned {status}")


def cpu_time():
    times = (resource.getrusage(scope) for scope in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return sum(usage.ru_utime + usage.ru_stime for usage in times)


def overlap_seconds(encode_intervals, decode_intervals):
    # Each engine has one worker, so intervals do not overlap within one engine.
    return sum(
        max(0, min(enc_end, dec_end) - max(enc_start, dec_start))
        for enc_start, enc_end in encode_intervals
        for dec_start, dec_end in decode_intervals
    )


def trial(mode, chains, hops, hardware):
    encoder = "h264_nvenc" if hardware else "libx264"
    all_seeds = [seed(chain) for chain in range(chains)]
    completed = []
    with TemporaryDirectory() as temp:
        directory = Path(temp)
        origin = time.perf_counter()
        monitor = VideoTelemetry(hardware, origin)
        cpu_start = cpu_time()

        def encode_stage(operand):
            start = time.perf_counter()
            encoded = encode(encoder, operand)
            return encoded, (start - origin, time.perf_counter() - origin)

        def decode_stage(chain, hop, encoded):
            fingerprint = hashlib.sha256(encoded).digest()
            plan_id = fingerprint[0] % len(PLANS)
            stream = directory / f"{chain}_{hop}.h264"
            stream.write_bytes(prediction_stream(encoded, PLANS[plan_id]))
            start = time.perf_counter()
            reference, output = decode(stream, hardware)
            return reference, output, plan_id, fingerprint.hex(), (start - origin, time.perf_counter() - origin)

        with monitor:
            if mode == "serial":
                for chain, operand in enumerate(all_seeds):
                    for hop in range(hops):
                        encoded, enc_times = encode_stage(operand)
                        decoded = decode_stage(chain, hop, encoded)
                        completed.append((chain, hop, operand, enc_times, decoded))
                        operand = decoded[1]
            else:
                with ThreadPoolExecutor(max_workers=1) as enc, ThreadPoolExecutor(max_workers=1) as dec:
                    pending = {}
                    for chain, operand in enumerate(all_seeds):
                        pending[enc.submit(encode_stage, operand)] = ("encode", chain, 0, operand, None)
                    while pending:
                        ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in ready:
                            stage, chain, hop, operand, enc_times = pending.pop(future)
                            if stage == "encode":
                                encoded, enc_times = future.result()
                                next_task = dec.submit(decode_stage, chain, hop, encoded)
                                pending[next_task] = ("decode", chain, hop, operand, enc_times)
                            else:
                                decoded = future.result()
                                completed.append((chain, hop, operand, enc_times, decoded))
                                if hop + 1 < hops:
                                    next_task = enc.submit(encode_stage, decoded[1])
                                    pending[next_task] = ("encode", chain, hop + 1, decoded[1], None)
            elapsed = time.perf_counter() - origin
            used_cpu = cpu_time() - cpu_start

    jobs = []
    enc_intervals, dec_intervals = [], []
    encode_differences = 0
    for chain, hop, operand, enc_times, decoded in sorted(completed):
        reference, output, plan_id, stream_sha, dec_times = decoded
        if output != expected_result(reference, PLANS[plan_id]):
            raise AssertionError(f"{mode} chain {chain} hop {hop}: decoded output differs from arithmetic oracle")
        encode_differences += sum(a != b for a, b in zip(reference, operand))
        enc_intervals.append(enc_times)
        dec_intervals.append(dec_times)
        jobs.append([
            chain, hop, plan_id, stream_sha,
            hashlib.sha256(operand).hexdigest(), hashlib.sha256(output).hexdigest(),
            *[round(t * 1000, 3) for t in (*enc_times, *dec_times)],
        ])

    assert len(jobs) == chains * hops
    for previous, current in zip(jobs, jobs[1:]):
        if previous[0] == current[0]:
            assert previous[5] == current[4] and previous[9] <= current[6]
    assert all(job[7] <= job[8] for job in jobs)
    samples = monitor.samples
    summary = {
        "mode": mode,
        "seconds": round(elapsed, 4),
        "host_cpu_seconds": round(used_cpu, 4),
        "jobs": len(jobs),
        "overlapping_subprocess_seconds": round(overlap_seconds(enc_intervals, dec_intervals), 4),
        "encode_changed_bytes": encode_differences,
        "distinct_offset_plans": sorted({job[2] for job in jobs}),
        "nvml_observations": len(samples),
        "nvml_both_positive_observations": sum(s[1] > 0 and s[3] > 0 for s in samples),
        "nvml_peak_encoder_pct": max((s[1] for s in samples), default=None),
        "nvml_peak_decoder_pct": max((s[3] for s in samples), default=None),
        "nvml_sampling_periods_us": sorted({s[p] for s in samples for p in (2, 4)}),
        "nvml_error": monitor.error,
    }
    return {"summary": summary, "jobs": jobs, "nvml": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-trace", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.chains <= len(LEVELS) or args.hops < 2:
        parser.error("Use 2–8 chains and at least two hops")
    hardware = args.mode == "gpu"
    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).decode().strip() if hardware else None
    # Warm both codecs once and alternate measured order to reduce startup bias.
    trial("serial", 1, 1, hardware)
    trials = []
    for mode in ("serial", "parallel", "parallel", "serial"):
        report = trial(mode, args.chains, args.hops, hardware)
        trials.append(report)
        print("TRIAL=" + json.dumps(report["summary"], separators=(",", ":")), flush=True)
    first_jobs = [job[:6] for job in trials[0]["jobs"]]
    exact = all([job[:6] for job in report["jobs"]] == first_jobs for report in trials)
    serial, parallel = [
        statistics.mean(report["summary"]["seconds"] for report in trials if report["summary"]["mode"] == mode)
        for mode in ("serial", "parallel")
    ]
    summary = {
        "gpu": gpu,
        "chains": args.chains,
        "hops_per_chain": args.hops,
        "all_arithmetic_and_dependencies_verified": True,
        "identical_serial_and_parallel_routes_and_outputs": exact,
        "average_serial_seconds": round(serial, 4),
        "average_parallel_seconds": round(parallel, 4),
        "speedup": round(serial / parallel, 4),
        "frame_dimensions": [WIDTH, HEIGHT],
    }
    full = {
        "summary": summary, "trials": trials,
        "job_fields": ["chain", "hop", "plan", "encoded_sha256", "input_sha256", "output_sha256",
                       "encode_start_ms", "encode_end_ms", "decode_start_ms", "decode_end_ms"],
        "nvml_fields": ["at_ms", "encoder_pct", "encoder_period_us", "decoder_pct", "decoder_period_us"],
    }
    args.output.write_text(json.dumps(full, indent=2) + "\n")
    print("PIPELINE_RESULT=" + json.dumps(summary, separators=(",", ":")), flush=True)
    if args.emit_trace:
        # Avoid repeating identical hashes four times in the Colab output transfer.
        packed = zlib.compress(json.dumps(full, separators=(",", ":")).encode(), 9)
        for index in range(0, len(packed), 1200):
            print("TRACE_PART=" + base64.b64encode(packed[index:index + 1200]).decode(), flush=True)
    if not exact:
        raise AssertionError("Serial and parallel modes took different routes or produced different outputs")


if __name__ == "__main__":
    main()
