#!/usr/bin/env python3
"""Reproducible proof-of-concept: repurpose a PNG decoder's Up predictor
for bytewise addition and compare a CPU video codec with zlib.

This uses software decoding ONLY. No GPU device or hardware media engine is assumed.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from numba import njit


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    data = tag + payload
    return struct.pack('>I', len(payload)) + data + struct.pack('>I', zlib.crc32(data) & 0xFFFFFFFF)


def make_adder_png(a: bytes, b: bytes) -> bytes:
    """Construct PNG scanlines with filter None, then Up.
    PNG's standard Up decoder reconstructs output_row_2[x] =
    (A[x] + B[x]) mod 256; no application-level addition is performed.
    """
    assert len(a) == len(b)
    header = struct.pack('>IIBBBBB', len(a), 2, 8, 0, 0, 0, 0)
    scanlines = b'\x00' + a + b'\x02' + b
    return b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', header) + png_chunk(b'IDAT', zlib.compress(scanlines, 1)) + png_chunk(b'IEND', b'')


def bench(fn, repetitions: int) -> float:
    values = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(repetitions):
            fn()
        values.append((time.perf_counter() - t) * 1000 / repetitions)
    return round(statistics.median(values), 5)


def codec_demo() -> dict:
    rng = np.random.default_rng(20261115)
    width = 16384
    a = rng.integers(0, 256, width, dtype=np.uint8)
    b = rng.integers(0, 256, width, dtype=np.uint8)
    encoded = make_adder_png(a.tobytes(), b.tobytes())

    def decode():
        with Image.open(io.BytesIO(encoded)) as im:
            return im.tobytes()[width:]

    expected = np.add(a, b, dtype=np.uint8).tobytes()
    assert decode() == expected
    # One pass is a tiny workload; use repeated trials and report per-call medians.
    return {
        'task': '(A[i] + B[i]) mod 256 for a 16,384-element byte vector',
        'exact_result': True,
        'encoded_png_bytes': len(encoded),
        'numpy_direct_add_ms': bench(lambda: np.add(a, b, dtype=np.uint8), 2000),
        'png_decoder_implicit_add_ms': bench(decode, 200),
        'hardware_used': 'CPU software PNG decoder (Pillow PNG decoding); not GPU',
        'principle': 'PNG Up scanline predictor performs bytewise addition during decoding'
    }



@njit
def compiled_cpu_prefix_sum(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    for col in range(values.shape[1]):
        out[0, col] = values[0, col]
    for row in range(1, values.shape[0]):
        for col in range(values.shape[1]):
            out[row, col] = np.uint8((int(out[row-1, col]) + int(values[row, col])) & 255)
    return out


def running_sum_demo() -> dict:
    """PNG Up filter computes all columnwise prefix sums modulo 256."""
    rng = np.random.default_rng(123456)
    height, width = 32, 8192
    values = rng.integers(0, 256, (height,width), dtype=np.uint8)
    scanlines = b'\x00' + values[0].tobytes()
    for i in range(1,height):
        scanlines += b'\x02' + values[i].tobytes()
    header = struct.pack('>IIBBBBB',width,height,8,0,0,0,0)
    png = b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR',header) + png_chunk(b'IDAT',zlib.compress(scanlines,1)) + png_chunk(b'IEND', b'')
    def decode():
        with Image.open(io.BytesIO(png)) as im:
            return im.tobytes()
    # Explicit CPU alternative: equivalent vectorized cumulative sums.
    expected = np.cumsum(values.astype(np.uint16),axis=0).astype(np.uint8).tobytes()
    assert decode() == expected
    assert compiled_cpu_prefix_sum(values).tobytes() == expected  # compile & warm up
    return {
        'task': 'Compute 32 successive running sums for each of 8,192 columns (mod 256)',
        'exact_result': True,
        'additions_implicitly_performed':(height-1)*width,
        'numpy_direct_prefix_sums_ms': bench(lambda: np.cumsum(values, axis=0, dtype=np.uint8),200),
        'compiled_cpu_direct_prefix_sums_ms': bench(lambda: compiled_cpu_prefix_sum(values),250),
        'png_decoder_implicit_prefix_sums_ms': bench(decode,75),
        'hardware_used': 'CPU software PNG decoder; not GPU'
    }


def video_demo() -> dict:
    frames = []
    for i in range(16):
        im = Image.new('L', (512, 512), color=255)
        draw = ImageDraw.Draw(im)
        for row in range(8, 500, 21):
            draw.text((12, row), f'DOCUMENT INDEX   ID {i:04d}    SECTION {row:04d}    MACHINE {i*37 + row:06d}', fill=0)
        for x in range(0, 500, 97):
            draw.line((x+i, 0, x+i, 511), fill=230, width=1)
        frames.append(im.tobytes())
    raw = b''.join(frames)
    d = {'raw_bytes': len(raw), 'frames': 16, 'resolution': '512x512', 'verified_exact': False}

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        path_in, path_out, path_dec = tdp / 'input.raw', tdp / 'video.mkv', tdp / 'decoded.raw'
        path_in.write_bytes(raw)
        enc_cmd = ['ffmpeg','-nostdin','-y','-hide_banner','-loglevel','error',
                   '-f','rawvideo','-pixel_format','gray','-video_size','512x512','-framerate','30','-i',str(path_in),
                   '-frames:v','16','-c:v','ffv1','-level','3','-threads','1',str(path_out)]
        dec_cmd = ['ffmpeg','-nostdin','-y','-hide_banner','-loglevel','error',
                   '-i',str(path_out), '-frames:v','16','-f','rawvideo','-pix_fmt','gray',str(path_dec)]

        # zlib performs the same actual *useful* job: losslessly storing and retrieving data.
        times = []
        packed = b''
        for _ in range(5):
            t = time.perf_counter()
            packed = zlib.compress(raw, 1)
            times.append(1000 * (time.perf_counter() - t))
        d['zlib_encode_ms'] = round(statistics.median(times),3)
        times = []
        for _ in range(5):
            t = time.perf_counter()
            unpacked = zlib.decompress(packed)
            times.append(1000 * (time.perf_counter() - t))
        assert unpacked == raw
        d['zlib_decode_ms'] = round(statistics.median(times),3)
        d['zlib_bytes'] = len(packed)

        trials = []
        for _ in range(3):
            t = time.perf_counter()
            p = subprocess.run(enc_cmd, capture_output=True, text=True)
            trials.append((time.perf_counter() - t)*1000)
            if p.returncode: raise RuntimeError(p.stderr)
        d['ffv1_encode_ms'] = round(statistics.median(trials),3)
        trials = []
        for _ in range(3):
            t = time.perf_counter()
            p = subprocess.run(dec_cmd, capture_output=True, text=True)
            trials.append((time.perf_counter() - t)*1000)
            if p.returncode: raise RuntimeError(p.stderr)
        d['ffv1_decode_ms'] = round(statistics.median(trials),3)
        d['ffv1_bytes'] = path_out.stat().st_size
        d['verified_exact'] = path_dec.read_bytes() == raw
    d['hardware_used'] = 'CPU software video codec FFV1; not NVENC/VAAPI/QSV'
    d['timing_note'] = 'FFmpeg measurements include process startup and filesystem I/O; zlib times are in-memory'
    return d


def hardware_probe() -> dict:
    cmd = ['ffmpeg', '-nostdin','-hide_banner','-loglevel','error','-f','lavfi','-i',
           'color=c=black:s=64x64:r=1','-frames:v','1','-c:v','h264_nvenc','-f','null','-']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return {'nvenc_success':p.returncode==0, 'return_code':p.returncode,
            'diagnostic':p.stderr.strip()[:850] if p.returncode else 'Hardware encoder accessible'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('initial-software-results.json'))
    args = parser.parse_args()
    out = {'png_math': codec_demo(), 'png_running_sum': running_sum_demo(), 'video_data': video_demo(), 'hardware_probe': hardware_probe()}
    print(json.dumps(out, indent=2))
    args.output.write_text(json.dumps(out, indent=2) + '\n')
