"""Top-level demo script to simulate the full flow: KGC -> registration -> key agreement."""

from loguru import logger

from protocol import protocol_key_agreement
from ta import TrustedAuthority
from user import User
from utils import report_timers, start_timer, stop_timer

def run_demo():
    logger.info("CL2PAKA demo")
    start_timer("total_run")

    # NIST P-256 curve
    start_timer("TrustedAuthority.setup_total")
    ta = TrustedAuthority.setup("secp256r1")
    stop_timer("TrustedAuthority.setup_total")

    start_timer("User.create_SM")
    sm = User.create("SM_001", "secp256r1")
    stop_timer("User.create_SM")

    start_timer("User.create_SP")
    sp = User.create("SP_A", "secp256r1")
    stop_timer("User.create_SP")

    # Registration (secure channel simulated)
    start_timer("TA.generate_partial_private_SM")
    dk_sm, R_sm = ta.generate_partial_private(sm.ID, sm.T)
    stop_timer("TA.generate_partial_private_SM")
    sm.set_partial_private(dk_sm, R_sm)

    start_timer("TA.generate_partial_private_SP")
    dk_sp, R_sp = ta.generate_partial_private(sp.ID, sp.T)
    stop_timer("TA.generate_partial_private_SP")
    sp.set_partial_private(dk_sp, R_sp)

    start_timer("protocol_key_agreement_total")
    sk_sm, sk_sp = protocol_key_agreement(sm, sp, ta)
    stop_timer("protocol_key_agreement_total")

    assert sk_sm == sk_sp
    logger.success("Success: both parties derived identical session key")

    stop_timer("total_run")
    report_timers()


if __name__ == "__main__":
    run_demo()
