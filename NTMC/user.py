"""User (Smart Meter / Service Provider) class."""
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger
from tinyec.ec import Curve, Point

from utils import get_curve_and_order, start_timer, stop_timer


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
        """
        Factory method to create a new user with initial long-term secret values.
        """
        curve, q = get_curve_and_order(curve_name)

        # random long-term secret t (timed)
        start_timer("User:rand_t")
        t = secrets.randbelow(q - 1) + 1
        stop_timer("User:rand_t")

        # T = t * G (timed)
        start_timer("User:ec_mul_T")
        T = t * curve.g
        stop_timer("User:ec_mul_T")

        return cls(ID=ID, curve=curve, q=q, t=t, T=T)

    def set_partial_private(self, d: int, R: "Point") -> None:
        """
        Store the partial private key components provided by the Trusted Authority.
        """
        self.d = int(d) % self.q
        self.R = R
        logger.info(f"[{self.ID}] Received partial private key R=({R.x}, {R.y}), d=***hidden***")

    def long_term_components(self) -> Tuple[int, "Point", int, "Point"]:
        """
        Retrieve the user's complete long-term key material.
        """
        if self.d is None or self.R is None:
            raise RuntimeError("User not registered: missing partial private key")
        return self.t, self.T, self.d, self.R
