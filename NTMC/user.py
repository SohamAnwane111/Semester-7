"""User (Smart Meter / Service Provider) class."""

# user.py

import secrets
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger
from tinyec.ec import Curve, Point
from utils import get_curve_and_order, xor_bytes
from tinyec import ec as _ec


@dataclass
class User:
    ID: str
    curve: "Curve"
    q: int
    t: int
    T: "Point"
    d: Optional[int] = None
    R: Optional["Point"] = None

    @classmethod
    def create(cls, ID: str, curve_name: str = "secp256r1") -> "User":
        """Factory method to create a new user with initial long-term secret values."""
        curve, q = get_curve_and_order(curve_name)
        t = secrets.randbelow(q - 1) + 1
        T = t * curve.g
        return cls(ID=ID, curve=curve, q=q, t=t, T=T)

    def set_partial_private(self, d: int, R: "Point") -> None:
        """Store the partial private key components (deprecated direct version)."""
        self.d = int(d) % self.q
        self.R = R
        logger.info(f"[{self.ID}] Received partial private key R=({R.x}, {R.y}), d=***hidden***")

    def set_partial_private_from_bytes(self, dk_enc: bytes, Rk_enc: bytes, secret_xor_val: int) -> None:
        """POC: undo single-byte XOR, parse dk and Rk, and set self.d/self.R."""
        dk_bytes = xor_bytes(dk_enc, secret_xor_val)
        Rk_bytes = xor_bytes(Rk_enc, secret_xor_val)

        dk_len = (self.q.bit_length() + 7) // 8
        if len(dk_bytes) != dk_len:
            raise RuntimeError("Malformed dk length after decryption")
        dk = int.from_bytes(dk_bytes, "big") % self.q

        byte_len = (self.curve.field.p.bit_length() + 7) // 8
        if len(Rk_bytes) != 2 * byte_len:
            raise RuntimeError("Malformed Rk bytes length after decryption")
        x = int.from_bytes(Rk_bytes[:byte_len], "big")
        y = int.from_bytes(Rk_bytes[byte_len:], "big")
        Rk = _ec.Point(self.curve, x, y)

        self.d = dk
        self.R = Rk
        logger.info(f"[{self.ID}] Received (decrypted) partial private key R=({Rk.x}, {Rk.y}), d=***hidden***")

    def long_term_components(self) -> Tuple[int, "Point", int, "Point"]:
        """Retrieve the user's complete long-term key material."""
        if self.d is None or self.R is None:
            raise RuntimeError("User not registered: missing partial private key")
        return self.t, self.T, self.d, self.R
