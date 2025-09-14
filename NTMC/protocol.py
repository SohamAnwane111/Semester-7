"""Protocol execution: simulate message exchange and compute session key."""
import secrets
from typing import Tuple

from ta import TrustedAuthority
from user import User
from utils import H1_id_T_R, H2_many, H3_derive, point_to_bytes

from loguru import logger


def protocol_key_agreement(sm: "User", sp: "User", ta: "TrustedAuthority") -> Tuple[bytes, bytes]:
    """Run the CL2PAKA key agreement and return the derived session key bytes for both parties."""
    curve = ta.curve
    q = ta.q
    Ppub = ta.Ppub

    logger.info(f"[Protocol] Starting key agreement between {sm.ID} and {sp.ID}")

    # SM -> SP: (Ri, Mi)
    ai = secrets.randbelow(q - 1) + 1
    Mi = ai * curve.g
    assert sm.R is not None
    
    logger.info(f"[{sm.ID}] Generated ephemeral Mi=({Mi.x}, {Mi.y})")

    # SP -> SM: (Rj, Mj)
    bj = secrets.randbelow(q - 1) + 1
    Mj = bj * curve.g
    assert sp.R is not None
    
    logger.info(f"[{sp.ID}] Generated ephemeral Mj=({Mj.x}, {Mj.y})")

    id_i_b = sm.ID.encode()
    id_j_b = sp.ID.encode()

    hj = H1_id_T_R(id_j_b, sp.T, sp.R, curve, q)
    hi = H1_id_T_R(id_i_b, sm.T, sm.R, curve, q)

    lij = H2_many(
        id_i_b,
        id_j_b,
        point_to_bytes(sm.T, curve),
        point_to_bytes(sp.T, curve),
        point_to_bytes(sm.R, curve),
        point_to_bytes(sp.R, curve),
        point_to_bytes(Mi, curve),
        point_to_bytes(Mj, curve),
        q=q,
    )

    scalar_left_sm = ((lij * ai) + sm.t + sm.d) % q
    right_point_sm = (lij * Mj) + sp.T + sp.R + (hj * Ppub)
    Kij_point = scalar_left_sm * right_point_sm

    scalar_left_sp = ((lij * bj) + sp.t + sp.d) % q
    right_point_sp = (lij * Mi) + sm.T + sm.R + (hi * Ppub)
    Kji_point = scalar_left_sp * right_point_sp

    if Kij_point.x != Kji_point.x or Kij_point.y != Kji_point.y:
        raise RuntimeError("Shared secret mismatch: Kij != Kji")

    h3_input = (
        id_i_b + id_j_b +
        point_to_bytes(sm.T, curve) + point_to_bytes(sp.T, curve) +
        point_to_bytes(sm.R, curve) + point_to_bytes(sp.R, curve) +
        point_to_bytes(Mi, curve) + point_to_bytes(Mj, curve) +
        point_to_bytes(Kij_point, curve)
    )

    logger.info(f"[Protocol] Derived shared point K = ({Kij_point.x}, {Kij_point.y})")
    sk_bytes = H3_derive(h3_input, length=32)
    logger.info(f"[Protocol] Final session key derived: {sk_bytes.hex()}")
     
    return sk_bytes, sk_bytes
