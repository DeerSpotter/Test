#!/usr/bin/env python3
"""Compile and run the NVIDIA Optical Flow SDK probe against the installed driver."""

import argparse
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


SDK = "edb50da3cf849840d680249aa6dbef248ebce2ca"
URL = f"https://raw.githubusercontent.com/NVIDIA/NVIDIAOpticalFlowSDK/{SDK}/"
SOURCE = Path(__file__).with_suffix(".cpp")


def run(command):
    process = subprocess.run(command, capture_output=True, text=True, timeout=40)
    if process.returncode:
        raise RuntimeError(f"Exit {process.returncode}: {' '.join(command)}\n{process.stdout}{process.stderr}")
    return process.stdout.strip()


def probe():
    device = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    getattr(ctypes.CDLL("libnvidia-opticalflow.so.1"), "NvOFAPICreateInstanceCuda")
    cuda = ctypes.CDLL("libcuda.so.1")
    assert cuda
    matches = [Path(line.rsplit(maxsplit=1)[-1]) for line in Path("/proc/self/maps").read_text().splitlines()]
    driver = next(path for path in matches if path.is_absolute() and path.name.startswith("libcuda.so"))
    with TemporaryDirectory() as directory:
        work = Path(directory)
        hashes = {}
        for name in ("nvOpticalFlowCommon.h", "nvOpticalFlowCuda.h"):
            with urlopen(URL + name, timeout=20) as response:
                content = response.read()
            hashes[name] = hashlib.sha256(content).hexdigest()
            (work / name).write_bytes(content)
        executable = work / "known_shift"
        run(["g++", "-std=c++17", "-O2", "-I", str(work), "-I", "/usr/local/cuda/include",
             str(SOURCE), str(driver), "-ldl", "-o", str(executable)])
        measured = subprocess.run([str(executable)], capture_output=True, text=True, timeout=40)
        if measured.returncode not in (0, 1):
            raise RuntimeError(f"Optical flow exited {measured.returncode}: {measured.stdout}{measured.stderr}")
        check = json.loads(measured.stdout)
    return {"gpu_and_driver": device, "sdk_header_commit": SDK, "headers_sha256": hashes,
            "api": "NvOFAPICreateInstanceCuda → nvOFExecute", **check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("OPTICAL_FLOW_RESULT=" + json.dumps(result, separators=(",", ":")))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
