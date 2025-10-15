"""Trusted Authority (KGC) implementation: setup and partial private generation."""
import secrets
from dataclasses import dataclass
from typing import Tuple

from loguru import logger
from tinyec.ec import Curve, Point

from utils import get_curve_and_order, hash_to_int, point_to_bytes, start_timer, stop_timer


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

        # master secret generation (timed)
        start_timer("TA:rand_msk")
        x = secrets.randbelow(q - 1) + 1
        stop_timer("TA:rand_msk")

        # master public Ppub = x * P (timed)
        start_timer("TA:ec_mul_Ppub")
        Ppub = x * P
        stop_timer("TA:ec_mul_Ppub")

        return cls(curve=curve, q=q, generator=P, msk=x, Ppub=Ppub)

    def generate_partial_private(self, ID: str, Tk: "Point") -> Tuple[int, "Point"]:
        # random rk (timed)
        start_timer("TA:rand_rk")
        rk = secrets.randbelow(self.q - 1) + 1
        stop_timer("TA:rand_rk")

        # Rk = rk * G (timed)
        start_timer("TA:ec_mul_Rk")
        Rk = rk * self.generator
        stop_timer("TA:ec_mul_Rk")

        # prepare hash input (not timed separately)
        id_b = ID.encode()
        h2_input = id_b + point_to_bytes(Tk, self.curve) + point_to_bytes(Rk, self.curve)

        # hk = H1/H2 mapping (timed inside hash_to_int)
        start_timer("TA:hash_hk")
        hk = hash_to_int(h2_input, self.q)
        stop_timer("TA:hash_hk")

        # combine to get dk
        start_timer("TA:compute_dk")
        dk = (rk + (hk * self.msk)) % self.q
        stop_timer("TA:compute_dk")

        logger.info(f"[TA] Generated partial private for {ID}: Rk=({Rk.x}, {Rk.y}), dk=***hidden***")
        return dk, Rk
