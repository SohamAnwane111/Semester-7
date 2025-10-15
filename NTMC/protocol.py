"""Protocol execution: simulate message exchange and compute session key."""
import secrets
from typing import Tuple

from loguru import logger

from ta import TrustedAuthority
from user import User
from utils import H1_id_T_R, H2_many, H3_derive, point_to_bytes, start_timer, stop_timer


def protocol_key_agreement(sm: "User", sp: "User", ta: "TrustedAuthority") -> Tuple[bytes, bytes]:
    """Run the CL2PAKA key agreement and return the derived session key bytes for both parties."""
    curve = ta.curve
    q = ta.q
    Ppub = ta.Ppub

    logger.info(f"[Protocol] Starting key agreement between {sm.ID} and {sp.ID}")

    # SM -> SP: (Ri, Mi)
    start_timer("protocol:SM_rand_ai")
    ai = secrets.randbelow(q - 1) + 1
    stop_timer("protocol:SM_rand_ai")

    start_timer("protocol:SM_ec_mul_Mi")
    Mi = ai * curve.g
    stop_timer("protocol:SM_ec_mul_Mi")

    assert sm.R is not None
    logger.info(f"[{sm.ID}] Generated ephemeral Mi=({Mi.x}, {Mi.y})")

    # SP -> SM: (Rj, Mj)
    start_timer("protocol:SP_rand_bj")
    bj = secrets.randbelow(q - 1) + 1
    stop_timer("protocol:SP_rand_bj")

    start_timer("protocol:SP_ec_mul_Mj")
    Mj = bj * curve.g
    stop_timer("protocol:SP_ec_mul_Mj")

    assert sp.R is not None
    logger.info(f"[{sp.ID}] Generated ephemeral Mj=({Mj.x}, {Mj.y})")

    id_i_b = sm.ID.encode()
    id_j_b = sp.ID.encode()

    # H1 computations (timed inside hash_to_int, but label the H1 calls here too)
    start_timer("protocol:H1_sp")
    hj = H1_id_T_R(id_j_b, sp.T, sp.R, curve, q)
    stop_timer("protocol:H1_sp")

    start_timer("protocol:H1_sm")
    hi = H1_id_T_R(id_i_b, sm.T, sm.R, curve, q)
    stop_timer("protocol:H1_sm")

    # H2 many pieces -> lij
    start_timer("protocol:H2_lij")
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
    stop_timer("protocol:H2_lij")

    start_timer("protocol:SM_compute_scalar_left")
    scalar_left_sm = ((lij * ai) + sm.t + sm.d) % q
    stop_timer("protocol:SM_compute_scalar_left")
    
    start_timer("protocol:SM_compute_right_point_lijMj")
    right_point_sm_part = (lij * Mj)
    stop_timer("protocol:SM_compute_right_point_lijMj")

    start_timer("protocol:SM_add_spT")
    right_point_sm = right_point_sm_part + sp.T + sp.R + (hj * Ppub)
    stop_timer("protocol:SM_add_spT")

    start_timer("protocol:SM_scalarmul_Kij")
    Kij_point = scalar_left_sm * right_point_sm
    stop_timer("protocol:SM_scalarmul_Kij")

    start_timer("protocol:SP_compute_scalar_left")
    scalar_left_sp = ((lij * bj) + sp.t + sp.d) % q
    stop_timer("protocol:SP_compute_scalar_left")

    start_timer("protocol:SP_compute_right_point_lijMi")
    right_point_sp_part = (lij * Mi)
    stop_timer("protocol:SP_compute_right_point_lijMi")

    start_timer("protocol:SP_add_smT")
    right_point_sp = right_point_sp_part + sm.T + sm.R + (hi * Ppub)
    stop_timer("protocol:SP_add_smT")

    start_timer("protocol:SP_scalarmul_Kji")
    Kji_point = scalar_left_sp * right_point_sp
    stop_timer("protocol:SP_scalarmul_Kji")

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
    start_timer("protocol:H3_derive")
    sk_bytes = H3_derive(h3_input, length=32)
    stop_timer("protocol:H3_derive")
    logger.info(f"[Protocol] Final session key derived: {sk_bytes.hex()}")

    return sk_bytes, sk_bytes
