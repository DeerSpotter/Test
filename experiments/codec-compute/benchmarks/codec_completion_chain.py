#!/usr/bin/env python3
"""Test codec-driven handoffs before input EOF on two concurrent streams."""

import argparse
import hashlib
import json
import os
import select
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from codec_two_input_speed import LENGTH, WIDTH, HEIGHT, checked, fixture, run


def commands(hardware):
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror"]
    output = ["-fps_mode", "passthrough", "-flush_packets", "1", "-f", "h264", "pipe:1"]
    raw_input = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
                 f"{WIDTH}x{HEIGHT}", "-r", "30", "-i", "pipe:0"]
    low_latency = ["-probesize", "1024", "-analyzeduration", "0"]
    if hardware:
        encoder = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ull",
                   "-rc", "constqp", "-qp", "1", "-bf", "0", "-g", "30", "-refs", "1",
                   "-forced-idr", "1", "-force_key_frames", "expr:1",
                   "-zerolatency", "1", "-delay", "0", "-aud", "1"]
        decoder = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                   "-c:v", "h264_cuvid"]
        mapped = ["-pix_fmt", "cuda"]
        host_copy = ["-vf", "hwdownload,format=nv12,format=yuv420p"]
    else:
        encoder = ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                   "-g", "30", "-qp", "1", "-profile:v", "main", "-force_key_frames",
                   "expr:1", "-x264-params", "weightp=1:ref=1:aud=1"]
        decoder = ["-c:v", "h264"]
        mapped, host_copy = [], []
    coded_input = low_latency + decoder + ["-f", "h264", "-i", "pipe:0"]
    # Verbose output confirms the middle stage kept decoded frames in CUDA format.
    middle = ["verbose" if item == "error" else item for item in base]
    return [
        base + raw_input + encoder + output,
        middle + coded_input + mapped + encoder + output,
        base + coded_input + host_copy + ["-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
               "-threads:v", "1", "-f", "rawvideo", "pipe:1"],
    ]


def pipeline(raw, hardware, chain):
    timeline = {}
    start = perf_counter()
    with TemporaryDirectory() as temp:
        logs = [open(Path(temp)/f"stage{i}.log","w+b") for i in range(3)]
        stages = []
        writer = None
        release = threading.Event()
        tail_started = threading.Event()
        problems = []
        try:
            source = subprocess.Popen(commands(hardware)[0], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=logs[0])
            stages.append(source)
            middle = subprocess.Popen(commands(hardware)[1], stdin=source.stdout,
                                      stdout=subprocess.PIPE, stderr=logs[1])
            stages.append(middle)
            source.stdout.close()
            sink = subprocess.Popen(commands(hardware)[2], stdin=middle.stdout,
                                    stdout=subprocess.PIPE, stderr=logs[2])
            stages.append(sink)
            middle.stdout.close()

            def produce():
                try:
                    with source.stdin:
                        boundary = len(raw) // (2*LENGTH) * LENGTH
                        source.stdin.write(raw[:boundary])
                        source.stdin.flush()
                        if not release.wait(40):
                            raise TimeoutError("No completed output while source input was open")
                        tail_started.set()
                        source.stdin.write(raw[boundary:])
                        timeline["tail_sent_ms"] = round(1000*(perf_counter()-start),3)
                except (OSError, TimeoutError) as error:
                    problems.append(str(error))

            writer=threading.Thread(target=produce)
            writer.start()
            received=bytearray()
            deadline=perf_counter()+45
            while True:
                wait=deadline-perf_counter()
                if wait <= 0 or not select.select([sink.stdout],[],[],wait)[0]:
                    raise TimeoutError("The codec stream stopped producing output")
                chunk=os.read(sink.stdout.fileno(),65536)
                if not chunk:
                    break
                received.extend(chunk)
                if len(received)>=LENGTH and not release.is_set():
                    timeline["first_complete_output_ms"] = round(1000*(perf_counter()-start),3)
                    timeline["first_output_before_second_input_half"] = not tail_started.is_set()
                    timeline["source_encoder_alive_on_first_output"] = source.poll() is None
                    release.set()
            writer.join(5)
            if writer.is_alive():
                raise TimeoutError("The source writer did not finish")
            exits = [p.wait(timeout=5) for p in stages]
            if any(exits) or problems:
                raise RuntimeError(f"Stage exits {exits}; input errors {problems}")
            checked("end-to-end frame bytes", received, raw)
            if not timeline.get("first_output_before_second_input_half") or not timeline["source_encoder_alive_on_first_output"]:
                raise AssertionError("The handoff did not make progress before the source finished")
            logs[1].seek(0)
            details=logs[1].read().decode(errors="replace")
            cuda_log=[line.strip() for line in details.splitlines() if (
                "pixfmt:cuda" in line or "Video: h264" in line and "cuda" in line)]
            if hardware and not cuda_log:
                raise AssertionError("Middle stage did not report CUDA surface format")
            return {"chain":chain,"frames":len(raw)//LENGTH,"bytes_exact":True,
                    "sha256":hashlib.sha256(received).hexdigest(),"stage_exit_codes":exits,
                    "gpu_surface_evidence":cuda_log,"elapsed_ms":round(1000*(perf_counter()-start),3),
                    **timeline}
        except Exception as error:
            for proc in stages:
                if proc.poll() is None:
                    proc.kill()
            for proc in stages:
                proc.wait()
            diagnostics=[]
            for index,logger in enumerate(logs):
                logger.seek(0)
                diagnostics.append(f"Stage {index}:\n"+logger.read().decode(errors="replace")[-6500:])
            raise RuntimeError(f"Chain {chain}: {error}\n"+"\n".join(diagnostics)) from error
        finally:
            release.set()
            if writer:
                writer.join(5)
            for proc in stages:
                if proc.stdout:
                    proc.stdout.close()
            for logger in logs:
                logger.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware",action="store_true")
    parser.add_argument("--frames",type=int,default=128)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.frames < 32 or args.frames % 2:
        parser.error("Use an even number of frames, at least 32")
    raw=fixture(args.frames//2+1)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures=[workers.submit(pipeline,raw[i*LENGTH:(i+args.frames)*LENGTH],args.hardware,i) for i in range(2)]
        results=[future.result() for future in futures]
    output={"hardware":args.hardware,"chains":results,"dispatch":"persistent FFmpeg stages connected by OS pipes"}
    if args.hardware:
        output["gpu"]=run(["nvidia-smi","--query-gpu=name,driver_version",
                           "--format=csv,noheader"]).decode().strip()
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print("HANDOFF_RESULT="+json.dumps(output,separators=(",",":")))


if __name__=="__main__":
    main()
