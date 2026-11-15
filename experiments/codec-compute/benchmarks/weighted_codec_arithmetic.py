#!/usr/bin/env python3
"""Encode an operand, then test H.264 weighted prediction as saturating addition."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


WIDTH = 256
HEIGHT = 256
LEVELS = (0, 16, 32, 64, 128, 192, 240, 255)
OFFSETS = (0, 19, -37, 63, -128, 127, -1, 1)
START = b"\x00\x00\x00\x01"
EPB = re.compile(rb"\x00\x00\x03(?=[\x00-\x03])")
FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror"]


def run(command, data=None):
    result = subprocess.run(command, input=data, capture_output=True, timeout=45)
    if result.returncode:
        diagnostic = result.stderr.decode(errors="replace")
        raise RuntimeError(f"Exit {result.returncode}: {' '.join(command)}\n{diagnostic}")
    return result.stdout


def ue(value):
    digits = f"{value + 1:b}"
    return "0" * (len(digits) - 1) + digits


def se(value):
    return ue(-2 * value if value <= 0 else 2 * value - 1)


class BitReader:
    def __init__(self, data):
        rbsp = EPB.sub(b"\x00\x00", data)
        self.bits = "".join(f"{byte:08b}" for byte in rbsp)
        self.pos = 0

    def take(self, count):
        value = self.bits[self.pos:self.pos + count]
        if len(value) != count:
            raise ValueError("Truncated H.264 header")
        self.pos += count
        return int(value, 2)

    def ue(self):
        zeros = 0
        while self.take(1) == 0:
            zeros += 1
        return (1 << zeros) - 1 + (self.take(zeros) if zeros else 0)

    def se(self):
        code = self.ue()
        return (code + 1) // 2 if code % 2 else -(code // 2)


def nal(header, bits):
    # RBSP stop bit, byte alignment and Annex B emulation prevention.
    bits += "1"
    bits += "0" * (-len(bits) % 8)
    escaped = bytearray()
    zeros = 0
    for index in range(0, len(bits), 8):
        value = int(bits[index:index + 8], 2)
        if zeros == 2 and value <= 3:
            escaped.append(3)
            zeros = 0
        escaped.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return START + bytes([header]) + escaped


def reference_parameters(anchor):
    units = [unit for unit in re.split(rb"\x00\x00\x00?\x01", anchor) if unit]
    sps = [u for u in units if u[0] & 31 == 7]
    pps = [u for u in units if u[0] & 31 == 8]
    idr = [u for u in units if u[0] & 31 == 5]
    if len(sps) != 1 or len(pps) != 1 or not idr:
        raise ValueError("Expected one SPS, one PPS and an IDR reference")
    reader = BitReader(sps[0][1:])
    profile = reader.take(8)
    reader.take(16)  # Constraint flags and level.
    if profile != 77:
        raise ValueError(f"Expected H.264 Main profile (77), got {profile}")
    sps_id = reader.ue()
    frame_bits = reader.ue() + 4
    poc_type = reader.ue()
    poc_bits = reader.ue() + 4 if poc_type == 0 else 0
    if poc_type not in (0, 2):
        raise ValueError(f"Unsupported picture-order-count type {poc_type}")
    max_refs = reader.ue()
    reader.take(1)  # gaps_in_frame_num_value_allowed_flag
    mb_width = reader.ue() + 1
    mb_height = reader.ue() + 1
    progressive = reader.take(1)
    if (mb_width * 16, mb_height * 16) != (WIDTH, HEIGHT) or not progressive or max_refs < 1:
        raise ValueError("Reference must be progressive, with matching dimensions and at least one reference slot")
    original_pps = BitReader(pps[0][1:]).ue()
    return sps_id, 1 if original_pps == 0 else 0, frame_bits, poc_type, poc_bits


def prediction_stream(anchor, offsets):
    sps_id, pps_id, frame_bits, poc_type, poc_bits = reference_parameters(anchor)
    # New PPS. The encoder's SPS, PPS and entire IDR stay byte-for-byte intact.
    pps_bits = (
        ue(pps_id) + ue(sps_id) + "00"  # CAVLC, no bottom-field POC.
        + ue(0) + ue(0) + ue(0)  # One slice group and one default reference per list.
        + "100"  # Weighted P prediction on, weighted B prediction off.
        + se(0) + se(0) + se(0) + "100"  # Deblocking control present.
    )
    mb_width = WIDTH // 16
    macroblocks_per_slice = mb_width * (HEIGHT // 16 // len(offsets))
    slices = []
    for row, offset in enumerate(offsets):
        slice_bits = ue(row * macroblocks_per_slice) + ue(0) + ue(pps_id)  # P slice.
        slice_bits += f"{1:0{frame_bits}b}"
        if poc_type == 0:
            slice_bits += f"{2:0{poc_bits}b}"
        slice_bits += "00"  # Default reference count and order.
        slice_bits += ue(0) + ue(0)  # Luma/chroma log2 weight denominators.
        slice_bits += "1" + se(1) + se(offset) + "0"  # Luma weight 1, offset; identity chroma.
        slice_bits += "0"  # Default reference-picture marking.
        slice_bits += se(0) + ue(1)  # QP delta zero, deblocking disabled.
        slice_bits += ue(macroblocks_per_slice)  # All P_Skip: no residual or motion data.
        slices.append(nal(0x41, slice_bits))
    return anchor + nal(0x68, pps_bits) + b"".join(slices)


def encode(encoder, operand):
    encoder_args = {
        "libx264": ["-qp", "1", "-x264-params", "weightp=1:ref=1"],
        "h264_nvenc": ["-preset", "p4", "-rc", "constqp", "-qp", "1", "-zerolatency", "1"],
    }
    return run(FFMPEG + [
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", "30", "-i", "pipe:0", "-frames:v", "1", "-c:v", encoder,
        "-profile:v", "main", "-bf", "0", "-g", "30", "-refs", "1",
        *encoder_args[encoder], "-f", "h264", "pipe:1",
    ], operand)


def decode(path, hardware):
    if hardware:
        inputs = [
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", "h264_cuvid",
        ]
        filters = ["-vf", "hwdownload,format=nv12,format=yuv420p"]
    else:
        inputs = ["-c:v", "h264"]
        filters = []
    raw = run(FFMPEG + inputs + ["-f", "h264", "-i", str(path)] + filters + [
        "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
    ])
    frame_bytes = WIDTH * HEIGHT * 3 // 2
    if len(raw) != 2 * frame_bytes:
        raise AssertionError(f"Expected two decoded frames, got {len(raw)} bytes")
    return raw[:frame_bytes], raw[frame_bytes:]


def require_match(actual, expected, label):
    if actual != expected:
        mismatches = sum(a != b for a, b in zip(actual, expected))
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
        detail = "" if first is None else f", first at byte {first}: {actual[first]} vs {expected[first]}"
        raise AssertionError(f"{label}: {mismatches} mismatches, {len(actual)} vs {len(expected)} bytes{detail}")


def experiment(encoder, hardware, directory):
    luma_bytes = WIDTH * HEIGHT
    row = bytes(level for level in LEVELS for _ in range(WIDTH // len(LEVELS)))
    operand = row * HEIGHT + bytes([128]) * (luma_bytes // 2)
    anchor = encode(encoder, operand)
    stream = directory / f"{encoder}_weighted_add.h264"
    stream.write_bytes(prediction_stream(anchor, OFFSETS))
    reference, software = decode(stream, hardware=False)
    reference_luma = reference[:luma_bytes]
    rows_per_slice = HEIGHT // len(OFFSETS)
    sums = [value + OFFSETS[(index // WIDTH) // rows_per_slice] for index, value in enumerate(reference_luma)]
    expected = bytes(min(255, max(0, value)) for value in sums) + reference[luma_bytes:]
    require_match(software, expected, "Software H.264 reconstruction vs scalar oracle")
    control = directory / f"{encoder}_zero_control.h264"
    control.write_bytes(prediction_stream(anchor, (0,) * len(OFFSETS)))
    control_reference, control_output = decode(control, hardware=hardware)
    require_match(control_reference, reference, "Zero-offset reference")
    require_match(control_output, reference, "Zero-offset reconstruction")
    result = {
        "encoder": encoder,
        "decoder": "h264_cuvid (CUDA output required)" if hardware else "h264 (CPU)",
        "luma_values": luma_bytes,
        "signed_offsets_per_macroblock_row": OFFSETS,
        "anchor_input_mismatches": sum(a != b for a, b in zip(operand, reference)),
        "low_clips": sum(value < 0 for value in sums),
        "high_clips": sum(value > 255 for value in sums),
        "exact_match_including_chroma": True,
        "zero_offset_control_exact": True,
        "reference_sha256": hashlib.sha256(reference).hexdigest(),
        "result_sha256": hashlib.sha256(expected).hexdigest(),
        "stream_sha256": hashlib.sha256(stream.read_bytes()).hexdigest(),
    }
    if hardware:
        gpu_reference, gpu_result = decode(stream, hardware=True)
        require_match(gpu_reference, reference, "NVDEC reference vs software reference")
        require_match(gpu_result, expected, "NVDEC arithmetic vs scalar oracle")
        result["gpu_output_sha256"] = hashlib.sha256(gpu_result).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpu", "gpu", "decode-only"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        result = experiment(
            "h264_nvenc" if args.mode == "gpu" else "libx264",
            args.mode != "cpu", Path(temporary),
        )
    if args.mode != "cpu":
        result["gpu"] = run([
            "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
        ]).decode().strip()
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
