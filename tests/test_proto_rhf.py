import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.rhf import (
    build_meas_ctx,
    force_bias_kernel_r,
    energy_kernel_r,
)
from ad_afqmc_prototype.trial.rhf import RhfTrial
#from ad_afqmc import wavefunctions

def _rand_orthonormal(key: jax.Array, n: int, k: int) -> jax.Array:
    a = jax.random.normal(key, (n, k)) + 1.0j * jax.random.normal(key, (n, k))
    q, _ = jnp.linalg.qr(a)
    return q[:, :k]

#def prep_old(mo_coeff, h0, h1, chol):
#    norb = mo_coeff.shape[0]
#    nocc = mo_coeff.shape[1]
#
#    wave_data = {
#        "mo_coeff": mo_coeff,
#    }
#
#    ham_data = {
#        "h0": jnp.array(h0),
#        "h1": [h1,h1],
#        "chol": chol,
#    }
#
#    ham_data["rot_h1"] = wave_data["mo_coeff"].T.conj() @ ham_data["h1"][0]
#    ham_data["rot_chol"] = jnp.einsum(
#        "pi,gij->gpj",
#        wave_data["mo_coeff"].T.conj(),
#        ham_data["chol"].reshape(-1, norb, norb),
#    )
#
#    trial = wavefunctions.rhf(norb=norb, nelec=(nocc, nocc))
#
#    return trial, ham_data, wave_data

def test_rhf_against_ref():
    key = jax.random.PRNGKey(3)
    k1, k2, k3 = jax.random.split(key, 3)
    norb, nocc, n_chol = 5, 2, 4

    h1 = jax.random.normal(k1, shape=(norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    chol = jax.random.normal(k2, shape=(n_chol, norb, norb))
    chol = 0.5 * (chol + jnp.einsum("gpq->gqp", chol))
    h0 = 1.234

    ham = HamChol(
        basis="restricted",
        h0=jnp.asarray(h0),
        h1=h1,
        chol=chol,
    )

    c = _rand_orthonormal(k3, norb, nocc).astype(jnp.complex128)
    tr = RhfTrial(mo_coeff=c)
    ctx = build_meas_ctx(ham, tr)

    w = jax.random.normal(k1, shape=(norb, nocc))

    fb_r_new = force_bias_kernel_r(w, None, ctx, tr)
    e_r_new = energy_kernel_r(w, ham, ctx, tr)

    #trial, ham_data, wave_data = prep_old(c, h0, h1, chol)
    #fb_r_old = trial._calc_force_bias_restricted(w, ham_data, wave_data)
    #e_r_old = trial._calc_energy_restricted(w, ham_data, wave_data)

    fb_r_old = jnp.array([
        3.01180282-2.56739074e-16j,
        9.33653684+9.02056208e-16j,
        2.80171513+3.24393290e-16j,
        7.83718773+7.07767178e-16j,
    ])

    e_r_old = 4.877867389920629-3.4791331244847307e-15j

    assert e_r_new - e_r_old < 1e-12
    assert jnp.max(jnp.abs(fb_r_new - fb_r_old)) < 1e-8

test_rhf_against_ref()
