"""Trusted Authority (Multi-KGC version): handles multi-source setup and partial key generation.

This version preserves the original interface `generate_partial_private(ID, Tk, secret_xor)`
returning XOR-protected (dk_bytes, Rk_bytes) like the POC you provided, but internally
the dk and R are computed as sequential contributions from `num_kgc` KGCs.
"""
# ta.py

import secrets
from dataclasses import dataclass
from typing import Tuple, List, Any
from loguru import logger
from tinyec.ec import Point
from tinyec import ec as ec

from utils import get_curve_and_order, hash_to_int, point_to_bytes, int_to_bytes_be, xor_bytes


@dataclass
class TrustedAuthority:
    curve: "ec.Curve"
    q: int
    generator: "Point"
    x_list: List[int]  # list of x_i for each KGC
    Ppub_list: List["Point"]
    Ppub: "Point"
    num_kgc: int

    @classmethod
    def setup(cls, curve_name: str = "secp256r1", num_kgc: int = 3) -> "TrustedAuthority":
        curve, q = get_curve_and_order(curve_name)
        P = curve.g

        x_list = [secrets.randbelow(q - 1) + 1 for _ in range(num_kgc)]
        Ppub_list = [x_i * P for x_i in x_list]

        Ppub = None
        for p in Ppub_list:
            Ppub = p if Ppub is None else Ppub + p

        ta = cls(curve=curve, q=q, generator=P, x_list=x_list, Ppub_list=Ppub_list, Ppub=Ppub, num_kgc=num_kgc)
        logger.success(f"[TA] Initialized {num_kgc} sequential KGCs on curve {curve_name}")
        for i, Ppub_i in enumerate(Ppub_list, start=1):
            logger.info(f"[KGC{i}] Ppub{i}=({Ppub_i.x}, {Ppub_i.y})")
        logger.info(f"[System] Aggregated public key Ppub=({Ppub.x}, {Ppub.y})")
        return ta

    def generate_partial_private(self, ID: str, Tk: "Point", secret_xor: int) -> Tuple[bytes, bytes]:
        """
        Sequential multi-KGC generation of user's partial private key.
        Mirrors original single-TA output but d and R are aggregated across KGC_i:
            r_total = sum r_i
            R_total = sum R_i
            d_total = sum (r_i + h * x_i) = r_total + h * sum x_i = r_total + h * x
        For consistency with your POC, we return (dk_enc_bytes, Rk_enc_bytes), each XORed with secret_xor.
        """
        id_b = ID.encode()

        Rk_total = None
        dk_total = 0

        # Each sequential KGC contributes (r_ki, R_ki) and uses its x_i
        for i in range(self.num_kgc):
            x_i = self.x_list[i]
            r_ki = secrets.randbelow(self.q - 1) + 1
            R_ki = r_ki * self.generator

            # Note: In your original POC hk depended on final R; to keep same H2 semantics
            # we compute hk on the final R after aggregating all R_ki. So we temporarily store r_i and R_i.
            # But to avoid a two-phase network roundtrip in this API, we simulate sequential behavior:
            # aggregate R first (like your protocol description required), then compute hk using final R.
            Rk_total = R_ki if Rk_total is None else Rk_total + R_ki

        # Now compute final h based on aggregated Rk_total
        h2_input = id_b + point_to_bytes(Tk, self.curve) + point_to_bytes(Rk_total, self.curve)
        hk = hash_to_int(h2_input, self.q)

        # Recompute dk_total properly: we must reuse the r_i values.
        # Simpler approach: regenerate deterministic r_i sequence from stored seeds is required,
        # but because r_i were ephemeral and not stored above, we will do a correct two-pass:
        # First pass we collected R_ki (and thus implicitly r_ki via discrete log impossible to extract).
        # To implement exactly as sequential KGCs we need to store r_ki; so instead we re-run with storing.
        # Implement proper storing below.

        # Proper implementation with stored r_i:
        Rk_total = None
        dk_total = 0
        r_list = []

        for i in range(self.num_kgc):
            r_ki = secrets.randbelow(self.q - 1) + 1
            r_list.append(r_ki)
            R_ki = r_ki * self.generator
            Rk_total = R_ki if Rk_total is None else Rk_total + R_ki

        # Recompute hk with final Rk_total
        h2_input = id_b + point_to_bytes(Tk, self.curve) + point_to_bytes(Rk_total, self.curve)
        hk = hash_to_int(h2_input, self.q)

        for i in range(self.num_kgc):
            x_i = self.x_list[i]
            r_ki = r_list[i]
            d_ki = (r_ki + (hk * x_i)) % self.q
            dk_total = (dk_total + d_ki) % self.q

        # Serialize dk_total and Rk_total as in original POC
        dk_len = (self.q.bit_length() + 7) // 8
        dk_bytes = int_to_bytes_be(dk_total, dk_len)
        Rk_bytes = point_to_bytes(Rk_total, self.curve)

        dk_enc = xor_bytes(dk_bytes, secret_xor)
        Rk_enc = xor_bytes(Rk_bytes, secret_xor)

        logger.info(f"[TA] Generated multi-KGC partial private for {ID}: Rk=({Rk_total.x}, {Rk_total.y}), dk=***hidden***")
        return dk_enc, Rk_enc
