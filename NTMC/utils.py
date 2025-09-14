"""Utility functions: hashing, HKDF, point serialization, curve helpers.

This module centralizes low-level operations so other modules remain clean and focused.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Tuple

from tinyec import ec, registry

Curve = ec.Curve
Point = ec.Point


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


# ---------------- Hashing / HKDF ----------------

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def hash_to_int(data: bytes, q: int) -> int:
    """Map bytes -> integer mod q using SHA-256 as H1/H2."""
    digest = sha256(data)
    return int.from_bytes(digest, "big") % q


def int_to_bytes_be(value: int, length: int) -> bytes:
    return value.to_bytes(length, "big")


def point_to_bytes(pt: Point, curve: Curve) -> bytes:
    """Serialize point to fixed-length x||y bytes (big-endian)."""
    p_bits = curve.field.p.bit_length()
    byte_len = (p_bits + 7) // 8
    return int_to_bytes_be(pt.x, byte_len) + int_to_bytes_be(pt.y, byte_len)


# HKDF utilities (HMAC-SHA256)

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if salt is None or len(salt) == 0:
        salt = bytes([0] * hashlib.sha256().digest_size)
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    hash_len = hashlib.sha256().digest_size
    blocks = (length + hash_len - 1) // hash_len
    okm = b""
    previous = b""
    for i in range(1, blocks + 1):
        previous = hmac.new(prk, previous + info + bytes([i]), hashlib.sha256).digest()
        okm += previous
    return okm[:length]


def hkdf(ikm: bytes, salt: bytes = b"", info: bytes = b"", length: int = 32) -> bytes:
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)


# High-level wrappers for the paper's hash functions

def H1_id_T_R(id_bytes: bytes, T: Point, R: Point, curve: Curve, q: int) -> int:
    data = id_bytes + point_to_bytes(T, curve) + point_to_bytes(R, curve)
    return hash_to_int(data, q)


def H2_many(*pieces: bytes, q: int) -> int:
    data = b"".join(pieces)
    return hash_to_int(data, q)


def H3_derive(*pieces: bytes, length: int = 32) -> bytes:
    ikm = b"".join(pieces)
    return hkdf(ikm, salt=b"", info=b"CL2PAKA-session", length=length)
