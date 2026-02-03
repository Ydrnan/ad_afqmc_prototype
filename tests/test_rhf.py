from ad_afqmc_prototype import config
from ad_afqmc_prototype.prop.types import QmcParams
import pytest
config.setup_jax()
import jax.numpy as jnp

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.trial.rhf import RhfTrial, make_rhf_trial_ops
from ad_afqmc_prototype.core.ops import MeasOps
from ad_afqmc_prototype.meas import rhf
from ad_afqmc_prototype.prep.pyscf_interface import get_integrals, get_trial_coeff
from ad_afqmc_prototype.prop.blocks import block
from ad_afqmc_prototype.prop.afqmc import make_prop_ops
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops
from ad_afqmc_prototype.core.ops import MeasOps
from ad_afqmc_prototype.meas import uhf
from ad_afqmc_prototype import driver
from pyscf import gto, scf

def run_calc(sys,meas_ops,ham_data,trial_ops,trial_data,params,block_fn,prop_ops):
    mean, err, block_e_all, block_w_all = driver.run_qmc_energy(
            sys=sys,
            params=params,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
        )
    return mean, err, block_e_all, block_w_all

def test_rhf_walker_rhf_hamiltonian(mf,params):
        meas_ops =  MeasOps(
                overlap=rhf.overlap_r,
                build_meas_ctx=rhf.build_meas_ctx,
                kernels={k_force_bias: rhf.force_bias_kernel_rw_rh, k_energy: rhf.energy_kernel_rw_rh},
            )
        h0, h1, chol = get_integrals(mf)
        sys = System(norb=mf.mol.nao, nelec=mf.mol.nelec, walker_kind="restricted")
        ham_data = HamChol(h0, h1, chol)
        mo = jnp.array(get_trial_coeff(mf))
        mo = mo[:,:sys.nup]
        trial_data = RhfTrial(mo)
        trial_ops = make_rhf_trial_ops(sys=sys)
        prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
        block_fn = block
        mean, err, block_e_all, block_w_all = run_calc(sys,meas_ops,ham_data,trial_ops,trial_data,params,block_fn,prop_ops) 
        assert jnp.isclose(mean, -108.69082190102914)
        assert jnp.isclose(err, 0.009301054598808593)

def test_uhf_walker_rhf_hamiltonian(mf,params):
        meas_ops =  MeasOps(
              overlap=rhf.overlap_u,
              build_meas_ctx=rhf.build_meas_ctx,
              kernels={k_force_bias: rhf.force_bias_kernel_uw_rh, k_energy: rhf.energy_kernel_uw_rh},
          )
        h0, h1, chol = get_integrals(mf)
        sys = System(norb=mf.mol.nao, nelec=mf.mol.nelec, walker_kind="unrestricted")
        ham_data = HamChol(h0, h1, chol)
        mo = jnp.array(get_trial_coeff(mf))
        mo_alpha = mo[:,:sys.nelec[0]]
        mo_beta = mo[:, :sys.nelec[1]]
        trial_data = RhfTrial(mo_alpha)
        trial_ops = make_rhf_trial_ops(sys=sys)
        prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
        block_fn = block
        mean, err, block_e_all, block_w_all = run_calc(sys,meas_ops,ham_data,trial_ops,trial_data,params,block_fn,prop_ops) 
        assert jnp.isclose(mean, -108.69082190102924)
        assert jnp.isclose(err, 0.009301054598809451)

@pytest.fixture(scope="module")
def mf():
    mol = gto.M(
        atom="""
        N 0.0000000 0.0000000 0.0000000
        N 0.0000000 0.0000000 1.1000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=10,
        n_blocks=100,
        seed=1234,
        n_walkers=20,
    )
      

if __name__ == "__main__":
    pytest.main([__file__])

