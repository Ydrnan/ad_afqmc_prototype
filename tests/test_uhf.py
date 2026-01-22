from ad_afqmc_prototype import config

config.setup_jax()

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.uhf import make_uhf_meas_ops
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops

def _rand_orthonormal_cols(key, nrow, ncol, dtype=jnp.complex128):
    """
    Random (nrow, ncol) matrix with orthonormal columns via QR.
    """
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(
        k1, (nrow, ncol), dtype=jnp.float64
    ) + 1.0j * jax.random.normal(k2, (nrow, ncol), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(a, mode="reduced")
    return q.astype(dtype)


def _make_random_ham_chol(key, norb, n_chol, dtype=jnp.float64) -> HamChol:
    """
    Build a small 'restricted' HamChol with:
      - symmetric real h1
      - symmetric real chol[g]
    """
    k1, k2, k3 = jax.random.split(key, 3)

    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)

    b = jax.random.normal(k2, (n_chol, norb, norb), dtype=dtype)
    chol = 0.5 * (b + jnp.swapaxes(b, 1, 2))

    h0 = jax.random.normal(k3, (), dtype=dtype)

    return HamChol(basis="restricted", h0=h0, h1=h1, chol=chol)

def _make_walkers(key, sys: System, dtype=jnp.complex128):
    norb, nup, ndn = sys.norb, sys.nup, sys.ndn
    wk = sys.walker_kind.lower()
    
    if wk == "restricted":
        if nup != ndn:
            raise ValueError("restricted walker tests require nup==ndn")
        # restricted walker is (norb, nocc) where nocc = nup = ndn
        return _rand_orthonormal_cols(key, norb, nup, dtype=dtype)
    
    if wk == "unrestricted":
        k1, k2 = jax.random.split(key)
        wu = _rand_orthonormal_cols(k1, norb, nup, dtype=dtype)
        wd = _rand_orthonormal_cols(k2, norb, ndn, dtype=dtype)
        return (wu, wd)
    
    if wk == "generalized":
        # generalized walker is (2*norb, nelec_total)
        return _rand_orthonormal_cols(key, 2 * norb, nup + ndn, dtype=dtype)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")

def _make_uhf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ca = _rand_orthonormal_cols(key, norb, nup, dtype=dtype)
    key, subkey = jax.random.split(key)
    cb = _rand_orthonormal_cols(subkey, norb, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("unrestricted", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)

    key = jax.random.PRNGKey(0)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_uhf_trial(k_trial, norb=norb, nup=nup, ndn=ndn)

    t_ops = make_uhf_trial_ops(sys)
    meas_manual = make_uhf_meas_ops(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = _make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)

    key = jax.random.PRNGKey(1)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_uhf_trial(k_trial, norb=norb, nup=nup, ndn=ndn)

    t_ops = make_uhf_trial_ops(sys)
    meas_manual = make_uhf_meas_ops(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    # Some implementations may not define energy for some walker kinds; skip in that case.
    if not meas_manual.has_kernel(k_energy):
        pytest.skip(
            f"manual UHF meas does not provide '{k_energy}' for walker_kind={walker_kind}"
        )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = _make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        print(jnp.max(jnp.abs(em-ea)))
        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)

if __name__ == "__main__":
    pytest.main([__file__])
