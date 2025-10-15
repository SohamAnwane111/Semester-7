"""Utility functions: hashing, HKDF, point serialization, curve helpers.

This module centralizes low-level operations so other modules remain clean and focused.
It also provides a lightweight timing API used across the codebase.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Tuple

from tinyec import ec, registry

Curve = ec.Curve
Point = ec.Point

_TIMERS = {}        
_TIMER_STARTS = {}  

def start_timer(name: str) -> None:
    """Start or resume a named timer."""
    _T = time.perf_counter()
    _TIMER_STARTS[name] = _T
    # ensure key exists
    if name not in _TIMERS:
        _TIMERS[name] = 0.0

def stop_timer(name: str) -> None:
    """Stop a named timer and accumulate elapsed time."""
    if name not in _TIMER_STARTS:
        # nothing to stop
        return
    now = time.perf_counter()
    start = _TIMER_STARTS.pop(name)
    elapsed = now - start
    _TIMERS[name] = _TIMERS.get(name, 0.0) + elapsed

def report_timers() -> None:
    """Print a timing summary for all named timers."""
    total = sum(_TIMERS.values())
    print("\n==== Computation Timing Report ====")
    for k, v in sorted(_TIMERS.items(), key=lambda x: x[0]):
        print(f"{k:40s}: {v:.6f} s")
    print(f"{'TOTAL (sum of labeled times)':40s}: {total:.6f} s")
    print("===================================\n")

def reset_timers() -> None:
    _TIMERS.clear()
    _TIMER_STARTS.clear()


# ----- Curve helpers -----
# Loads an elliptic curve and its order.
def get_curve_and_order(name: str = "secp256r1") -> Tuple[Curve, int]:
    """Return a tinyec curve and its order q."""
    curve = registry.get_curve(name)
    possible_attrs = ("n", "order", "r")
    q = None
    for attr in possible_attrs:
        if hasattr(curve, attr):
            q = getattr(curve, attr)
            break
    if q is None and hasattr(curve, "field"):
        for attr in possible_attrs:
            if hasattr(curve.field, attr):
                q = getattr(curve.field, attr)
                break
    if q is None:
        raise RuntimeError("Unable to determine curve order (q). Check tinyec version.")
    return curve, int(q)

# Hashing and mapping to integers.
def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

# Hashing and mapping to integers (timed)
def hash_to_int(data: bytes, q: int) -> int:
    """Map bytes -> integer mod q using SHA-256 as H1/H2."""
    start_timer("hash_sha256")
    digest = sha256(data)
    stop_timer("hash_sha256")
    return int.from_bytes(digest, "big") % q

def int_to_bytes_be(value: int, length: int) -> bytes:
    return value.to_bytes(length, "big")

def point_to_bytes(pt: Point, curve: Curve) -> bytes:
    """Serialize point to fixed-length x||y bytes (big-endian)."""
    p_bits = curve.field.p.bit_length()
    byte_len = (p_bits + 7) // 8
    return int_to_bytes_be(pt.x, byte_len) + int_to_bytes_be(pt.y, byte_len)


# HKDF utilities (HMAC-SHA256) — timed
# HMAC-based Key Derivation Function.

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    start_timer("hkdf_extract")
    if salt is None or len(salt) == 0:
        salt = bytes([0] * hashlib.sha256().digest_size)
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    stop_timer("hkdf_extract")
    return prk


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    start_timer("hkdf_expand")
    hash_len = hashlib.sha256().digest_size
    blocks = (length + hash_len - 1) // hash_len
    okm = b""
    previous = b""
    for i in range(1, blocks + 1):
        previous = hmac.new(prk, previous + info + bytes([i]), hashlib.sha256).digest()
        okm += previous
    stop_timer("hkdf_expand")
    return okm[:length]


def hkdf(ikm: bytes, salt: bytes = b"", info: bytes = b"", length: int = 32) -> bytes:
    # hkdf_extract and hkdf_expand already timed
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)


# High-level wrappers for the paper's hash functions

def H1_id_T_R(id_bytes: bytes, T: Point, R: Point, curve: Curve, q: int) -> int:
    # It takes the user’s identity + their public key + the KGC’s partial key, and turns it into a number.
    data = id_bytes + point_to_bytes(T, curve) + point_to_bytes(R, curve)
    # hash_to_int is timed
    return hash_to_int(data, q)

def H2_many(*pieces: bytes, q: int) -> int:
    # identities, long-term keys, and the random one-time keys (Mi, Mj) — then hashes them all into one number.
    data = b"".join(pieces)
    return hash_to_int(data, q)

def H3_derive(*pieces: bytes, length: int = 32) -> bytes:
    # After both parties compute the same shared ECC point, H3 turns it into a 256-bit random session key using HKDF.
    ikm = b"".join(pieces)
    start_timer("H3_hkdf_total")
    res = hkdf(ikm, salt=b"", info=b"CLs2PAKA-session", length=length)
    stop_timer("H3_hkdf_total")
    return res
