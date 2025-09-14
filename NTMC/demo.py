"""Top-level demo script to simulate the full flow: KGC -> registration -> key agreement."""

from loguru import logger

from protocol import protocol_key_agreement
from ta import TrustedAuthority
from user import User


def run_demo():
    logger.info("CL2PAKA demo")
    ta = TrustedAuthority.setup("secp256r1")
    sm = User.create("SM_001", "secp256r1")
    sp = User.create("SP_A", "secp256r1")

    # Registration (secure channel simulated)
    dk_sm, R_sm = ta.generate_partial_private(sm.ID, sm.T)
    sm.set_partial_private(dk_sm, R_sm)

    dk_sp, R_sp = ta.generate_partial_private(sp.ID, sp.T)
    sp.set_partial_private(dk_sp, R_sp)

    sk_sm, sk_sp = protocol_key_agreement(sm, sp, ta)
    assert sk_sm == sk_sp
    logger.success("Success: both parties derived identical session key")


if __name__ == "__main__":
    run_demo()
