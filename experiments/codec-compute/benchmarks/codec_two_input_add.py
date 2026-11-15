#!/usr/bin/env python3
"""Add two independently populated byte arrays with one H.264 B picture."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from codec_batch_benchmark import AUD, run
from weighted_codec_arithmetic import BitReader, FFMPEG, HEIGHT, WIDTH, nal, reference_parameters, se, ue


LUMA = WIDTH * HEIGHT
LENGTH = LUMA * 3 // 2
VALUES = (0, 1, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 255)
START_CODE = re.compile(rb"\x00\x00\x00?\x01")


def operands():
    span = WIDTH // len(VALUES)
    left_row = bytes(value for value in VALUES for _ in range(span))
    left = left_row * HEIGHT
    right = b"".join(bytes([value]) * WIDTH * span for value in VALUES)
    chroma = bytes([128]) * (LUMA // 2)
    return left + chroma, right + chroma


def encode_references(first, second, hardware, force_intra=False):
    if hardware:
        options = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "constqp",
                   "-qp", "1", "-aud", "1", "-forced-idr", "0"]
    else:
        options = ["-c:v", "libx264", "-qp", "1", "-x264-params",
                   "aud=1:repeat-headers=1:b-adapt=0:scenecut=0"]
    if force_intra:
        options += ["-force_key_frames", "expr:eq(n,1)"]
    return run(FFMPEG + [
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", "30", "-i", "pipe:0", "-frames:v", "2", "-profile:v", "main",
        "-g", "30", "-bf", "1", "-refs", "2", *options, "-f", "h264", "pipe:1",
    ], first + second)


def units(access_unit):
    return [unit for unit in START_CODE.split(access_unit) if unit]


def read_vcl_header(unit, frame_bits, poc_type, poc_bits):
    reader = BitReader(unit[1:])
    first_mb = reader.ue()
    slice_type = reader.ue() % 5
    reader.ue()  # pic_parameter_set_id
    number = reader.take(frame_bits)
    if unit[0] & 31 == 5:
        reader.ue()  # idr_pic_id
    poc = reader.take(poc_bits) if poc_type == 0 else 2 * number
    return first_mb, slice_type, number, poc


def two_reference_stream(anchors, left_weight, right_weight):
    frames = [part for part in AUD.split(anchors) if part]
    if len(frames) != 2:
        raise ValueError(f"Expected two encoder access units, received {len(frames)}")
    sps_id, pps_id, frame_bits, poc_type, poc_bits = reference_parameters(frames[0])
    if poc_type != 0:
        raise ValueError(f"Expected picture order count type 0 for B-picture reordering, got {poc_type}")
    sps = next(u for u in units(frames[0]) if (u[0] & 31) == 7)
    sps_reader = BitReader(sps[1:])
    sps_reader.take(24)
    for _ in range(4):
        sps_reader.ue()
    max_refs = sps_reader.ue()
    if max_refs < 2:
        raise ValueError(f"Encoder SPS allows only {max_refs} decoded reference pictures")

    vcl_units = [
        [u for u in units(frame) if (u[0] & 31) in (1, 5)]
        for frame in frames
    ]
    if not all(vcl_units):
        raise ValueError("Both encoder frames must have a coded picture")
    if any((u[0] & 31) != 5 for u in vcl_units[0]) or any(
            (u[0] & 31) != 1 or (u[0] >> 5) == 0 for u in vcl_units[1]):
        raise ValueError("Expected an IDR followed by a non-IDR reference picture")
    headers = [read_vcl_header(frame[0], frame_bits, poc_type, poc_bits) for frame in vcl_units]
    a, b = headers
    if a[0] != 0 or b[0] != 0 or a[2] != 0 or b[2] != 1 or b[3] - a[3] < 2:
        raise ValueError(f"Unexpected encoder frame numbers or picture order counts: {headers}")

    # One default reference per list, explicit bidirectional weights, CAVLC.
    pps_bits = ue(pps_id) + ue(sps_id) + "00" + ue(0) * 3 + "001"
    pps_bits += se(0) * 3 + "100"  # Deblocking control present.
    poc = (a[3] + b[3]) // 2
    frame_num = (b[2] + 1) % (1 << frame_bits)
    bits = ue(0) + ue(1) + ue(pps_id) + f"{frame_num:0{frame_bits}b}"
    bits += f"{poc:0{poc_bits}b}"
    bits += "1000"  # Spatial direct, default counts, default L0/L1 order.
    bits += ue(0) * 2  # Luma and chroma weight denominators.
    for weight in (left_weight, right_weight):
        bits += "1" + se(weight) + se(0) + "0"  # Luma weight, no offset, identity chroma.
    bits += se(0) + ue(1)  # QP delta zero, deblocking disabled.
    # B_Bi_16x16, zero motion in both lists, zero coded residual.
    macroblock = ue(0) + ue(3) + se(0) * 4 + ue(0)
    bits += macroblock * (WIDTH * HEIGHT // 256)
    metadata = {"reference_picture_order_counts": [a[3], b[3]],
                "reference_slice_types": [a[1], b[1]], "result_picture_order_count": poc,
                "allowed_reference_pictures": max_refs}
    return anchors + nal(0x68, pps_bits) + nal(0x09, "010") + nal(0x01, bits), metadata


def decode(encoded, hardware, count):
    device = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
              "-c:v", "h264_cuvid"] if hardware else ["-c:v", "h264"]
    output = ["-vf", "hwdownload,format=nv12,format=yuv420p"] if hardware else []
    raw = run(FFMPEG + device + ["-f", "h264", "-i", "pipe:0"] + output + [
        "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-threads:v", "1",
        "-f", "rawvideo", "pipe:1",
    ], encoded)
    if len(raw) != count * LENGTH:
        raise AssertionError(f"Expected {count} frames, received {len(raw)} raw bytes")
    return [raw[i:i+LENGTH] for i in range(0, len(raw), LENGTH)]


def verify(label, actual, reference):
    if actual != reference:
        differences = sum(a != b for a, b in zip(actual, reference))
        first = next((i for i, (a, b) in enumerate(zip(actual, reference)) if a != b), None)
        detail = "" if first is None else f"; at byte {first}: got {actual[first]}, expected {reference[first]}"
        raise AssertionError(f"{label} differs at {differences} bytes{detail}")


def run_test(hardware, force_intra):
    a, b = operands()
    anchors = encode_references(a, b, hardware, force_intra)
    first_ref, second_ref = decode(anchors, hardware, 2)
    verify("First encoded operand", first_ref, a)
    verify("Second encoded operand", second_ref, b)
    summary = {}
    for label, weight in (("saturating_sum", 2), ("rounded_average_control", 1)):
        stream, metadata = two_reference_stream(anchors, weight, weight)
        first, result, second = decode(stream, hardware, 3)
        reference = bytes(min(255, (weight * (x + y) + 1) >> 1)
                          for x, y in zip(a[:LUMA], b[:LUMA], strict=True)) + a[LUMA:]
        verify(label, result, reference)
        verify("First operand in arithmetic stream", first, a)
        verify("Second operand in arithmetic stream", second, b)
        summary[label] = {
            "weights": [weight, weight], "luma_values": LUMA,
            "examples": {str((a[y*WIDTH+x], b[y*WIDTH+x])): result[y*WIDTH+x]
                         for x, y in ((16, 64), (64, 160), (240, 240), (0, 0))},
            "output_sha256": hashlib.sha256(result).hexdigest(),
            "stream_sha256": hashlib.sha256(stream).hexdigest(),
        }
    return {
        "encoder": "h264_nvenc" if hardware else "libx264",
        "decoder": "h264_cuvid with CUDA output" if hardware else "software h264",
        "both_encoder_inputs_exact": True,
        "both_b_picture_results_exact": True,
        "distinct_operand_pairs": len(VALUES) ** 2,
        "clipped_sums": sum(x+y > 255 for x, y in zip(a[:LUMA], b[:LUMA], strict=True)),
        "dimensions": [WIDTH, HEIGHT],
        **metadata, **summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--force-intra", action="store_true", help="Force the second encoder reference to be intra-coded")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_test(args.hardware, args.force_intra)
    if args.hardware:
        result["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version",
                             "--format=csv,noheader"]).decode().strip()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("TWO_INPUT_RESULT=" + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
