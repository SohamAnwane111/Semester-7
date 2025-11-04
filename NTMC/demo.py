"""Top-level demo script to simulate the full flow: KGC -> registration -> key agreement."""
from loguru import logger
from protocol import protocol_key_agreement
from ta import TrustedAuthority
from user import User
from config import secret_xor


def run_demo():
    logger.info("CL2PAKA demo (split KAs + key confirmation)")

    ta = TrustedAuthority.setup("secp256r1")
    sm = User.create("SM_001", "secp256r1")
    sp = User.create("SP_A", "secp256r1")

    # Registration using XOR-protected exchange
    dk_sm_enc, R_sm_enc = ta.generate_partial_private(sm.ID, sm.T)
    sm.set_partial_private_from_bytes(dk_sm_enc, R_sm_enc, secret_xor)

    dk_sp_enc, R_sp_enc = ta.generate_partial_private(sp.ID, sp.T)
    sp.set_partial_private_from_bytes(dk_sp_enc, R_sp_enc, secret_xor)

    # Execute split/sequential protocol
    sk_sm, sk_sp = protocol_key_agreement(sm, sp, ta)

    assert sk_sm == sk_sp
    logger.success("Success: both parties derived identical session key and performed key confirmation")


if __name__ == "__main__":
    run_demo()
