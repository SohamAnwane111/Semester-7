"""Trusted Authority (KGC) implementation: setup and partial private generation."""
import secrets
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
from tinyec.ec import Curve, Point

from utils import get_curve_and_order, hash_to_int, point_to_bytes


@dataclass
class TrustedAuthority:
    curve: "Curve"
    q: int
    generator: "Point"
    msk: int  # master secret x
    Ppub: "Point"

    @classmethod
    def setup(cls, curve_name: str = "secp256r1") -> "TrustedAuthority":
        curve, q = get_curve_and_order(curve_name)
        P = curve.g
        x = secrets.randbelow(q - 1) + 1
        Ppub = x * P
        return cls(curve=curve, q=q, generator=P, msk=x, Ppub=Ppub)

    def generate_partial_private(self, ID: str, Tk: "Point") -> Tuple[int, "Point"]:
        rk = secrets.randbelow(self.q - 1) + 1
        Rk = rk * self.generator
        id_b = ID.encode()
        h2_input = id_b + point_to_bytes(Tk, self.curve) + point_to_bytes(Rk, self.curve)
        hk = hash_to_int(h2_input, self.q)
        dk = (rk + (hk * self.msk)) % self.q
        logger.info(f"[TA] Generated partial private for {ID}: Rk=({Rk.x}, {Rk.y}), dk=***hidden***")
        return dk, Rk
