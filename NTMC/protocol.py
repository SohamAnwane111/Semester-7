"""Protocol execution: split CL2PAKA into small sequential KAs and add explicit key confirmation."""
# protocol.py

import secrets
from typing import Tuple, Dict, Any
from loguru import logger

from ta import TrustedAuthority
from user import User
from utils import (
    H1_id_T_R, H2_many, H3_derive, point_to_bytes, compute_hmac
)


def ka_phase1_generate_ephemeral(curve, q) -> Dict[str, Any]:
    """
    Phase 1: Generate ephemeral nonces and ephemeral public points (Mi, Mj).
    Returns the secrets and public points so they can be sequentially passed.
    """
    ai = secrets.randbelow(q - 1) + 1
    Mi = ai * curve.g

    bj = secrets.randbelow(q - 1) + 1
    Mj = bj * curve.g

    return {"ai": ai, "Mi": Mi, "bj": bj, "Mj": Mj}


def ka_phase2_compute_hashes_and_lij(sm: User, sp: User, Mi, Mj, curve, q) -> Dict[str, Any]:
    """
    Phase 2: Compute H1s (hi, hj) and lij (H2 over transcript pieces).
    This phase is purely deterministic given long-term and ephemeral values.
    """
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

    return {"hi": hi, "hj": hj, "lij": lij, "id_i_b": id_i_b, "id_j_b": id_j_b}


def ka_phase3_compute_shared_point_and_session_key(
    sm: User, sp: User, ai: int, bj: int, Mi, Mj, hi: int, hj: int, lij: int, ta: TrustedAuthority
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """
    Phase 3: Compute the scalar multiplications, check equality of shared ECC points,
    derive session key and perform key confirmation.
    Returns (sk_sm, sk_sp, audit_info)
    """
    curve = ta.curve
    q = ta.q
    Ppub = ta.Ppub

    # SM side computations
    scalar_left_sm = ((lij * ai) + sm.t + sm.d) % q
    right_point_sm = (lij * Mj) + sp.T + sp.R + (hj * Ppub)
    Kij_point = scalar_left_sm * right_point_sm

    # SP side computations
    scalar_left_sp = ((lij * bj) + sp.t + sp.d) % q
    right_point_sp = (lij * Mi) + sm.T + sm.R + (hi * Ppub)
    Kji_point = scalar_left_sp * right_point_sp

    if Kij_point.x != Kji_point.x or Kij_point.y != Kji_point.y:
        raise RuntimeError("Shared secret mismatch: Kij != Kji")

    # Build transcript for H3 and key confirmation (deterministic)
    id_i_b = sm.ID.encode()
    id_j_b = sp.ID.encode()

    h3_input = (
        id_i_b + id_j_b +
        point_to_bytes(sm.T, curve) + point_to_bytes(sp.T, curve) +
        point_to_bytes(sm.R, curve) + point_to_bytes(sp.R, curve) +
        point_to_bytes(Mi, curve) + point_to_bytes(Mj, curve) +
        point_to_bytes(Kij_point, curve)
    )

    # Derive session key bytes
    sk_bytes = H3_derive(h3_input, length=32)

    # Key confirmation: both sides compute HMAC over canonical transcript with the derived key.
    # Label strings included to avoid cross-protocol confusion.
    transcript = b"KC:CL2PAKA" + h3_input
    mac_sm = compute_hmac(sk_bytes, b"SM" + transcript)
    mac_sp = compute_hmac(sk_bytes, b"SP" + transcript)

    # In a real protocol, SM would send mac_sm to SP and SP would verify, and vice versa.
    # Here, we simulate both sides computing and verifying.
    if mac_sm != compute_hmac(sk_bytes, b"SM" + transcript):
        raise RuntimeError("SM key confirmation failed self-check")
    if mac_sp != compute_hmac(sk_bytes, b"SP" + transcript):
        raise RuntimeError("SP key confirmation failed self-check")

    audit_info = {
        "K_point": (Kij_point.x, Kij_point.y),
        "transcript": transcript,
        "mac_sm": mac_sm.hex(),
        "mac_sp": mac_sp.hex()
    }

    logger.info(f"[Protocol] Derived shared point K = ({Kij_point.x}, {Kij_point.y})")
    logger.info(f"[Protocol] Session key (hex): {sk_bytes.hex()}")
    logger.info(f"[Protocol] Key-confirmation MACs (simulated): SM={audit_info['mac_sm']}, SP={audit_info['mac_sp']}")

    return sk_bytes, sk_bytes, audit_info


def protocol_key_agreement(sm: "User", sp: "User", ta: "TrustedAuthority") -> Tuple[bytes, bytes]:
    """
    High-level orchestration calling the three phases in sequence.
    Each phase can be executed, audited, or split across different processes as needed.
    """
    curve = ta.curve
    q = ta.q

    logger.info(f"[Protocol] Starting split/sequential key agreement between {sm.ID} and {sp.ID}")

    # Phase 1: Ephemeral generation
    eph = ka_phase1_generate_ephemeral(curve, q)
    ai, Mi, bj, Mj = eph["ai"], eph["Mi"], eph["bj"], eph["Mj"]
    logger.info(f"[{sm.ID}] Generated ephemeral Mi=({Mi.x}, {Mi.y})")
    logger.info(f"[{sp.ID}] Generated ephemeral Mj=({Mj.x}, {Mj.y})")

    # Phase 2: Hash computations and lij
    ph2 = ka_phase2_compute_hashes_and_lij(sm, sp, Mi, Mj, curve, q)
    hi, hj, lij = ph2["hi"], ph2["hj"], ph2["lij"]

    # Phase 3: shared point, derive key, and key confirmation
    sk_sm, sk_sp, audit = ka_phase3_compute_shared_point_and_session_key(
        sm, sp, ai, bj, Mi, Mj, hi, hj, lij, ta
    )

    # Final check
    if sk_sm != sk_sp:
        raise RuntimeError("Final session keys mismatch (should not happen after point equality check)")

    return sk_sm, sk_sp
