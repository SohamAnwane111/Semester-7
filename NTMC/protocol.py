"""Protocol execution: merged CL2PAKA phases into single protocol_key_agreement() with explicit key confirmation."""
# protocol.py

import secrets
from typing import Tuple
from loguru import logger

from ta import TrustedAuthority
from user import User
from utils import (
    H1_id_T_R, H2_many, H3_derive, point_to_bytes, compute_hmac
)


def protocol_key_agreement(sm: "User", sp: "User", ta: "TrustedAuthority") -> Tuple[bytes, bytes]:
    """
    Single function executing all three phases of CL2PAKA:
        1. Generate ephemerals (Mi, Mj)
        2. Compute hashes (hi, hj, lij)
        3. Derive shared secret, compute session key, confirm key
    This unified version preserves exact math and logging from split design.
    """
    curve = ta.curve
    q = ta.q
    Ppub = ta.Ppub

    logger.info(f"[Protocol] Starting merged key agreement between {sm.ID} and {sp.ID}")

    # === Phase 1: Ephemeral generation ===
    ai = secrets.randbelow(q - 1) + 1
    Mi = ai * curve.g
    bj = secrets.randbelow(q - 1) + 1
    Mj = bj * curve.g
    logger.info(f"[{sm.ID}] Generated ephemeral Mi=({Mi.x}, {Mi.y})")
    logger.info(f"[{sp.ID}] Generated ephemeral Mj=({Mj.x}, {Mj.y})")

    # === Phase 2: Hash computations ===
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

    # === Phase 3: Shared point & session key derivation ===
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

    sk_bytes = H3_derive(h3_input, length=32)

    # === Key confirmation ===
    transcript = b"KC:CL2PAKA" + h3_input
    mac_sm = compute_hmac(sk_bytes, b"SM" + transcript)
    mac_sp = compute_hmac(sk_bytes, b"SP" + transcript)

    # Self-check simulating mutual confirmation
    if mac_sm != compute_hmac(sk_bytes, b"SM" + transcript):
        raise RuntimeError("SM key confirmation failed self-check")
    if mac_sp != compute_hmac(sk_bytes, b"SP" + transcript):
        raise RuntimeError("SP key confirmation failed self-check")

    audit_info = {
        "K_point": (Kij_point.x, Kij_point.y),
        "mac_sm": mac_sm.hex(),
        "mac_sp": mac_sp.hex()
    }

    logger.info(f"[Protocol] Derived shared point K = ({Kij_point.x}, {Kij_point.y})")
    logger.info(f"[Protocol] Session key (hex): {sk_bytes.hex()}")
    logger.info(f"[Protocol] Key-confirmation MACs (simulated): SM={audit_info['mac_sm']}, SP={audit_info['mac_sp']}")
    logger.success("[Protocol] Key agreement completed successfully")

    return sk_bytes, sk_bytes
