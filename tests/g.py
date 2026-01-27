from ad_afqmc import afqmc
from ad_afqmc import wavefunctions, utils

import jax
import jax.numpy as jnp

from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.ucisd import make_ucisd_meas_ops
from ad_afqmc_prototype.trial.ucisd import UcisdTrial, make_ucisd_trial_ops
from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype import testing

from ad_afqmc_prototype import config

from pyscf import gto, scf, cc
import scipy.linalg as la

config.setup_jax()

class Obj:
    pass

mol = gto.M(atom="""
    H 0.0 0.0 0.0
    H 1.5 0.0 0.0
    H 4.0 0.5 0.0
    H 5.5 0.0 0.0
    """,
    basis = "6-31g")

# UHF
mf = scf.UHF(mol)
mf.kernel()
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)
mo1 = mf.stability()[0]
mf = mf.newton().run(mo1, mf.mo_occ)

# UHF -> GHF
gmf = scf.addons.convert_to_ghf(mf)

# UCCSD
ucc = cc.CCSD(mf)
ucc.kernel()

tmp_ghf = "tmp_ghf"
tmp_uhf = "tmp_uhf"
chol_cut = 1e-5
# UCCSD -> GCCSD
gcc = cc.addons.convert_to_gccsd(ucc)
jnp.savez(tmp_ghf+"/amplitudes.npz",t1=gcc.t1, t2=gcc.t2)

na, nb = mol.nelec
nelec = na+nb
nmo = mf.mo_coeff.shape[-1]
norb = nmo

options = {}

# gcisd
utils.prep_afqmc_ghf_complex(mol, gmf, tmpdir=tmp_ghf, chol_cut=chol_cut)
gcisd = Obj()
gcisd.options = options
gcisd.options["trial"] = "gcisd_complex"
gcisd.options["walker_type"] = "generalized"
gcisd.ham_data, gcisd.ham, gcisd.prop, gcisd.trial, gcisd.wave_data, gcisd.sampler, gcisd.observable, gcisd.options = utils.setup_afqmc(
    options=gcisd.options, tmp_dir=tmp_ghf
)

# ucisd
utils.prep_afqmc(ucc, chol_cut=chol_cut, tmpdir=tmp_uhf, write_to_disk=True)
ucisd = Obj()
ucisd.options = options
ucisd.options["trial"] = "ucisd"
ucisd.options["walker_type"] = "unrestricted"
ucisd.ham_data, ucisd.ham, ucisd.prop, ucisd.trial, ucisd.wave_data, ucisd.sampler, ucisd.observable, ucisd.options = utils.setup_afqmc(
    options=ucisd.options, tmp_dir=tmp_uhf
)
ucisd.ham_data = ucisd.trial._build_measurement_intermediates(ucisd.ham_data, ucisd.wave_data)

# w_ghf <-> w_uhf
# w_uhf = Y.T @ X @ w_ghf
# w_ghf = X.T @ Y @ w_uhf
# with X and Y s.t.
# uhf_mos X = ghf_mos
# uhf_trial Y = ghf_trial

# Unitary transformation UHF MOs <-> GHF MOs
uhf_mos = la.block_diag(mf.mo_coeff[0], mf.mo_coeff[1])
ghf_mos = gmf.mo_coeff
# uhf_mos X = ghf_mos
X = jnp.linalg.solve(uhf_mos, ghf_mos)

# Check
# X.T @ X = Id
res = jnp.linalg.norm(X.T @ X - jnp.identity(2*nmo))
assert res < 1e-14, res

# Unitary transformation UHF trial <-> GHF trial
uhf_trial = la.block_diag(ucisd.wave_data["mo_coeff"][0], ucisd.wave_data["mo_coeff"][1])
ghf_trial = gcisd.wave_data["mo_coeff"][0]
# uhf_trial Y = ghf_trial
Y = jnp.linalg.solve(uhf_trial, ghf_trial)

# Check
# Y.T @ Y = Id
res = jnp.linalg.norm(Y.T @ Y - jnp.identity(2*nmo))
assert res < 1e-14

# Define UHF and GHF trials for the phase
uhf_a = ucisd.wave_data["mo_coeff"][0][:,:na]
uhf_b = ucisd.wave_data["mo_coeff"][1][:,:nb]
uhf_trial = la.block_diag(uhf_a, uhf_b)
ghf_trial = gcisd.wave_data["mo_coeff"][0][:,:nelec]

def assert_ghf(w_ghf):
    res = jnp.linalg.norm(w_ghf - X.T @ Y @ Y.T @ X @ w_ghf)
    assert res < 1e-14

def assert_uhf(w_uhf):
    res = jnp.linalg.norm(w_uhf - Y.T @ X @ X.T @ Y @ w_uhf)
    assert res < 1e-14

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
wa = testing.rand_orthonormal_cols(k1, norb, na)
wb = testing.rand_orthonormal_cols(k2, norb, nb)
w = testing.rand_orthonormal_cols(k2, 2*norb, nelec)

assert_ghf(w)
assert_uhf(w)
w_ghf = w
w_uhf = Y.T @ X @ w_ghf
e1 = ucisd.trial._calc_energy_generalized(w_uhf, ucisd.ham_data, ucisd.wave_data)
e2 = gcisd.trial._calc_energy_generalized(w, gcisd.ham_data, gcisd.wave_data)
print(e1)
print(e2)
#fb1 = ucisd.trial._calc_force_bias_generalized(w, ucisd.ham_data, ucisd.wave_data)
fb2 = gcisd.trial._calc_force_bias_generalized(w, gcisd.ham_data, gcisd.wave_data)
#print(fb1)
print(fb2)

n_oa, n_ob = na, nb
n_va = norb - n_oa
n_vb = norb - n_ob

walker_kind="generalized"
sys = System(norb=norb, nelec=nelec, walker_kind=walker_kind)

ham = HamChol(
    basis="restricted",
    h0=ucisd.ham_data["h0"],
    h1=ucisd.ham_data["h1"][0],
    chol=ucisd.ham_data["chol"].reshape(-1, norb, norb)
)

trial = UcisdTrial(
    mo_coeff_a=jnp.array(ucisd.wave_data["mo_coeff"][0]),
    mo_coeff_b=jnp.array(ucisd.wave_data["mo_coeff"][1]),
    c1a=ucisd.wave_data["ci1A"],
    c1b=ucisd.wave_data["ci1B"],
    c2aa=ucisd.wave_data["ci2AA"],
    c2ab=ucisd.wave_data["ci2AB"],
    c2bb=ucisd.wave_data["ci2BB"],
)

t_ops = make_ucisd_trial_ops(sys)
meas_manual = make_ucisd_meas_ops(sys)
meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

ctx_manual = meas_manual.build_meas_ctx(ham, trial)
ctx_auto = meas_auto.build_meas_ctx(ham, trial)

e_manual = meas_manual.require_kernel(k_energy)
e_auto = meas_auto.require_kernel(k_energy)

fb_manual = meas_manual.require_kernel(k_force_bias)
fb_auto = meas_auto.require_kernel(k_force_bias)

e3 = e_manual(w_uhf, ham, ctx_manual, trial)
e4 = e_auto(w_uhf, ham, ctx_auto, trial)

print(e3)
print(e4)

fb3 = fb_manual(w_uhf, ham, ctx_manual, trial)
fb4 = fb_auto(w_uhf, ham, ctx_auto, trial)

print(fb3)
print(fb4)
