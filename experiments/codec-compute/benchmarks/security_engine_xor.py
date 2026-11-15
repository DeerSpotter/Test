#!/usr/bin/env python3
"""Verify a batched bitwise XOR mapping onto AES-CBC decryption."""

import argparse
import json
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path


BLOCK_SIZE = 16
REQUEST_BYTES = 4096
TEST_KEY = bytes(range(BLOCK_SIZE))
ZERO_IV = bytes(BLOCK_SIZE)
TEGRA_DRIVERS = {"ecb": "ecb-aes-tegra", "cbc": "cbc-aes-tegra"}


def openssl_transform(mode, data, decrypt=False):
    operation = "-d" if decrypt else "-e"
    command = ["openssl", "enc", f"-aes-128-{mode}", operation, "-nopad", "-K", TEST_KEY.hex()]
    if mode == "cbc":
        command.extend(["-iv", ZERO_IV.hex()])
    process = subprocess.run(command, input=data, capture_output=True, timeout=30)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace").strip())
    if len(process.stdout) != len(data):
        raise RuntimeError(f"OpenSSL {mode} returned {len(process.stdout)} of {len(data)} bytes")
    return process.stdout


def tegra_transform(mode, data, decrypt=False):
    if not hasattr(socket, "AF_ALG") or not hasattr(socket.socket, "sendmsg_afalg"):
        raise RuntimeError("The Jetson backend requires Linux Python with AF_ALG support")
    driver = TEGRA_DRIVERS[mode]
    operation = socket.ALG_OP_DECRYPT if decrypt else socket.ALG_OP_ENCRYPT
    result = bytearray()
    try:
        with socket.socket(socket.AF_ALG, socket.SOCK_SEQPACKET) as cipher:
            cipher.bind(("skcipher", driver))
            cipher.setsockopt(socket.SOL_ALG, socket.ALG_SET_KEY, TEST_KEY)
            for offset in range(0, len(data), REQUEST_BYTES):
                chunk = data[offset:offset + REQUEST_BYTES]
                with cipher.accept()[0] as request:
                    request.settimeout(10)
                    if mode == "cbc":
                        written = request.sendmsg_afalg([chunk], op=operation, iv=ZERO_IV)
                    else:
                        written = request.sendmsg_afalg([chunk], op=operation)
                    if written != len(chunk):
                        raise RuntimeError(f"{driver} accepted {written} of {len(chunk)} bytes")
                    output = request.recv(len(chunk))
                    if len(output) != len(chunk):
                        raise RuntimeError(f"{driver} returned {len(output)} of {len(chunk)} bytes")
                # Requests restart with a zero IV. Each starts with the predecessor
                # block for a pair, so the following result is independent of that IV.
                result.extend(output)
    except OSError as error:
        raise RuntimeError(f"AF_ALG driver {driver}: {error}") from error
    return bytes(result)


def prepare_left(transform, left):
    """Encrypt each left operand as an independent 16-byte block."""
    if not left or len(left) % BLOCK_SIZE:
        raise ValueError("Left operands must contain complete 16-byte blocks")
    return transform("ecb", left)


def xor_prepared(transform, encrypted_left, right):
    """Interleave each right operand and encrypted left operand for CBC decoding."""
    if len(encrypted_left) != len(right) or not right or len(right) % BLOCK_SIZE:
        raise ValueError("Both operands must have the same number of complete 16-byte blocks")
    ciphertext = bytearray()
    for offset in range(0, len(right), BLOCK_SIZE):
        ciphertext.extend(right[offset:offset + BLOCK_SIZE])
        ciphertext.extend(encrypted_left[offset:offset + BLOCK_SIZE])
    plaintext = transform("cbc", ciphertext, decrypt=True)
    # Each even-positioned output (zero based) belongs to a predecessor block.
    return b"".join(
        plaintext[offset:offset + BLOCK_SIZE]
        for offset in range(BLOCK_SIZE, len(plaintext), 2 * BLOCK_SIZE)
    )


def verify(backend, pairs):
    transform = tegra_transform if backend == "jetson" else openssl_transform
    generator = random.Random(20261115)
    size = pairs * BLOCK_SIZE
    left = generator.randbytes(size)
    right = generator.randbytes(size)
    prepared = prepare_left(transform, left)
    checks = []
    for name, right_operand in (("random", right), ("reused_left_with_zero", bytes(size)),
                                ("reused_left_with_ones", b"\xff" * size)):
        expected = bytes(a ^ b for a, b in zip(left, right_operand))
        actual = xor_prepared(transform, prepared, right_operand)
        exact = sum(
            actual[offset:offset + BLOCK_SIZE] == expected[offset:offset + BLOCK_SIZE]
            for offset in range(0, size, BLOCK_SIZE)
        )
        checks.append({"case": name, "exact_pairs": exact, "passed": exact == pairs})
    result = {
        "backend": backend,
        "machine": platform.machine(),
        "block_bytes": BLOCK_SIZE,
        "pairs_per_case": pairs,
        "kernel_release": platform.release(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    if backend == "jetson":
        result["kernel_drivers"] = TEGRA_DRIVERS
        result["request_bytes"] = REQUEST_BYTES
    else:
        result["openssl_version"] = subprocess.check_output(["openssl", "version"], text=True).strip()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("openssl", "jetson"), default="openssl")
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    try:
        result = verify(args.backend, args.pairs)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 2
    report = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
