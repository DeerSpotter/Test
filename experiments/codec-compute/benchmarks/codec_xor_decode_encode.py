#!/usr/bin/env python3
"""Compute XOR in H.264 reference pictures and re-encode the result."""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path
from time import perf_counter

from codec_two_input_add import LENGTH, LUMA, verify
from weighted_codec_arithmetic import FFMPEG, HEIGHT, WIDTH, nal, run, se, ue


PAIRS = 4096
BIT_PLANES = tuple(bytes(255 if value & (1 << bit) else 0 for value in range(256))
                   for bit in range(8))
NEUTRAL_CHROMA = bytes([128]) * (LUMA // 2)
PCM_BITS = tuple(f"{value:08b}" for value in range(256))


def parameter_sets():
    # Main profile, level 3.1, four reference slots, POC type 0 with five bits.
    sps = f"{77:08b}{0:08b}{31:08b}" + ue(0) * 3 + ue(1) + ue(4) + "0"
    sps += ue(WIDTH // 16 - 1) + ue(HEIGHT // 16 - 1) + "1100"
    pps = ue(0) * 2 + "00" + ue(0) * 3 + "001" + se(0) * 3 + "100"
    return nal(0x67, sps) + nal(0x68, pps)


def pcm_reference(plane, is_idr):
    poc = 0 if is_idr else 8
    frame_num = 0 if is_idr else 1
    bits = ue(0) + ue(2) + ue(0) + f"{frame_num:04b}"
    if is_idr:
        bits += ue(0)
    bits += f"{poc:05b}" + ("00" if is_idr else "0") + se(0) + ue(1)
    parts = []
    for y in range(0, HEIGHT, 16):
        for x in range(0, WIDTH, 16):
            prefix = (bits if not parts else "") + ue(25)  # I_PCM macroblock.
            prefix += "0" * (-len(prefix) % 8)
            start = y * WIDTH + x
            block = b"".join(plane[row:row + 16] for row in range(start, start + 16 * WIDTH, WIDTH))
            parts.append(prefix + "".join(PCM_BITS[value] for value in block) + PCM_BITS[128] * 128)
    kind = 0x65 if is_idr else 0x41
    return nal(0x09, "000") + nal(kind, "".join(parts))


def weighted_picture(frame_num, poc, weights, reference, use_first_anchor=False):
    bits = ue(0) + ue(1) + ue(0) + f"{frame_num:04b}{poc:05b}"
    bits += "10"  # Spatial direct, one reference per list.
    if use_first_anchor:
        # Current frame_num is 3. Select frame_num 0 (difference 3) for L0.
        bits += "1" + ue(0) + ue(2) + ue(3)
    else:
        bits += "0"
    bits += "0" + ue(0) * 2  # Default L1, zero weight denominators.
    for weight in weights:
        bits += "1" + se(weight) + se(0) + "0"
    if reference:
        bits += "0"  # Default reference picture marking.
    bits += se(0) + ue(1)
    macroblock = ue(0) + ue(3) + se(0) * 4 + ue(0)  # B_Bi_16x16.
    bits += macroblock * (WIDTH * HEIGHT // 256)
    return nal(0x09, "010") + nal(0x41 if reference else 0x01, bits)


def program_xor(left_plane, right_plane):
    return b"".join((
        parameter_sets(),
        pcm_reference(left_plane, True),
        pcm_reference(right_plane, False),
        weighted_picture(2, 2, (2, -2), True),
        weighted_picture(3, 6, (-2, 2), True, use_first_anchor=True),
        weighted_picture(4, 4, (2, 2), False),
    ))


def transcode(stream, hardware):
    decoder = ["-c:v", "h264"]
    encoder = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0"]
    if hardware:
        decoder = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid"]
        encoder = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "lossless",
                   "-profile:v", "high444p", "-rc", "constqp", "-qp", "0", "-pix_fmt", "cuda"]
    command = FFMPEG.copy()
    command[command.index("error")] = "verbose"
    command += decoder + ["-f", "h264", "-i", "pipe:0",
                          "-vf", r"select=eq(mod(n\,5)\,2)"]
    command += encoder + ["-bf", "0", "-fps_mode", "passthrough", "-f", "h264", "pipe:1"]
    start = perf_counter()
    completed = subprocess.run(command, input=stream, capture_output=True, timeout=90)
    elapsed = perf_counter() - start
    logs = completed.stderr.decode(errors="replace")
    if completed.returncode:
        raise RuntimeError(f"Decode and encode exited {completed.returncode}\n{logs}")
    cuda_input = [line.strip() for line in logs.splitlines() if "pixfmt:cuda" in line]
    if hardware and not cuda_input:
        raise AssertionError(f"No CUDA frame handoff in FFmpeg output\n{logs}")
    return completed.stdout, cuda_input, elapsed


def verify_test(hardware):
    prior_fixture = random.Random(20261115)
    left = prior_fixture.randbytes(PAIRS * 16)
    random_right = prior_fixture.randbytes(len(left))
    cases = (("random", random_right), ("reused_left_with_zero", bytes(len(left))),
             ("reused_left_with_ones", b"\xff" * len(left)))
    if len(left) != LUMA or WIDTH % 16 or HEIGHT % 16:
        raise AssertionError("The fixture requires one operand byte per luma position")
    cpu_runs = []
    for _ in range(7):
        cpu_start = perf_counter()
        direct_outputs = tuple(bytes(a ^ b for a, b in zip(left, right, strict=True))
                               for _, right in cases)
        cpu_runs.append((perf_counter() - cpu_start) * 1000)
    groups = []
    start = perf_counter()
    for _, right in cases:
        for table in BIT_PLANES:
            groups.append(program_xor(left.translate(table), right.translate(table)))
    stream = b"".join(groups)
    preparation = perf_counter() - start
    encoded, cuda_input, elapsed = transcode(stream, hardware)
    decode_start = perf_counter()
    verified = run(FFMPEG + ["-c:v", "h264", "-f", "h264", "-i", "pipe:0",
                             "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
                             "-threads:v", "1", "-f", "rawvideo", "pipe:1"], encoded)
    output_decode_ms = (perf_counter() - decode_start) * 1000
    reconstruction_start = perf_counter()
    reconstructed = []
    for case_index in range(len(cases)):
        output = bytearray(LUMA)
        for bit in range(len(BIT_PLANES)):
            offset = (case_index * len(BIT_PLANES) + bit) * LENGTH
            for position, sample in enumerate(verified[offset:offset + LUMA]):
                if sample == 255:
                    output[position] |= 1 << bit
                elif sample != 0:
                    raise AssertionError("Nonbinary output in XOR bit-plane")
        reconstructed.append(bytes(output))
    reconstruction_ms = (perf_counter() - reconstruction_start) * 1000
    raw_input_to_bytes_ms = (perf_counter() - start) * 1000
    for (name, _), actual, direct in zip(cases, reconstructed, direct_outputs, strict=True):
        verify(f"Reconstructed XOR bytes: {name}", actual, direct)
    expected = [direct.translate(table) + NEUTRAL_CHROMA
                for direct in direct_outputs for table in BIT_PLANES]
    verify("Decoded pixels of the re-encoded XOR planes", verified, b"".join(expected))
    if len(verified) != len(expected) * LENGTH:
        raise AssertionError("Unexpected encoded frame count")
    result = {
        "hardware": hardware,
        "decoder": "h264_cuvid" if hardware else "h264",
        "encoder": "h264_nvenc" if hardware else "libx264",
        "verification_decoder": "h264 (CPU)",
        "dimensions": [WIDTH, HEIGHT],
        "verified_bit_planes": len(expected),
        "verified_xor_bits": len(expected) * LUMA,
        "exact_16_byte_pairs": {name: PAIRS for name, _ in cases},
        "output_layout": "one luma plane per bit, LSB first",
        "ffmpeg_cuda_surface_evidence": cuda_input,
        "prepared_h264_bytes": len(stream),
        "encoded_h264_bytes": len(encoded),
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "verified_pixels_sha256": hashlib.sha256(verified).hexdigest(),
        "reconstructed_bytes_sha256": hashlib.sha256(b"".join(reconstructed)).hexdigest(),
        "preparation_ms": round(preparation * 1000, 3),
        "decode_then_encode_wall_ms": round(elapsed * 1000, 3),
        "output_decode_ms": round(output_decode_ms, 3),
        "byte_reconstruction_ms": round(reconstruction_ms, 3),
        "raw_input_to_bytes_ms": round(raw_input_to_bytes_ms, 3),
        "direct_python_xor_median_ms": round(statistics.median(cpu_runs), 3),
        "direct_python_xor_runs_ms": [round(value, 3) for value in cpu_runs],
    }
    if hardware:
        result["gpu_and_driver"] = run(["nvidia-smi", "--query-gpu=name,driver_version",
                                        "--format=csv,noheader"]).decode().strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify_test(args.hardware)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("CODEC_XOR_RESULT=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
