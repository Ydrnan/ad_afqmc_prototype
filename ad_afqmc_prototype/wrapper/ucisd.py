import numpy as np
import jax.numpy as jnp

from .. import driver, config
from ..prep import integrals
from ..core.system import System
from ..ham.chol import HamChol
from ..prop.afqmc import make_prop_ops
from ..prop.blocks import block
from ..prop.types import QmcParams
from ..prep.pyscf_interface import get_integrals, get_ucisd, get_trial_coeff
from ..trial.ucisd import UcisdTrial, make_ucisd_trial_ops
from ..meas.ucisd import make_ucisd_meas_ops

class Ucisd:
    def __init__(self, cc):
        config.setup_jax()

        mol = cc.mol
        mf = cc._scf
        h0, h1, chol = get_integrals(mf)
        c1a, c1b, c2aa, c2ab, c2bb = get_ucisd(cc)

        sys = System(norb=mol.nao, nelec=mol.nelec, walker_kind="unrestricted")
        ham_data = HamChol(h0, h1, chol)

        moa, mob = get_trial_coeff(mf)

        self.trial_data = UcisdTrial(
            mo_coeff_a=moa,
            mo_coeff_b=mob,
            c1a=c1a,
            c1b=c1b,
            c2aa=c2aa,
            c2ab=c2ab,
            c2bb=c2bb,
        )
        self.trial_ops = make_ucisd_trial_ops(sys=sys)
        self.meas_ops = make_ucisd_meas_ops(sys=sys)
        self.prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
        self.params = QmcParams(
            n_eql_blocks=20, n_blocks=200, seed=np.random.randint(0, int(1e6))
        )
        self.block_fn = block
        self.sys = sys
        self.ham_data = ham_data

    def kernel(self):
        return driver.run_qmc_energy(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
        )
