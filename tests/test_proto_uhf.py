import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.uhf import (
    build_meas_ctx,
    force_bias_kernel_u,
    energy_kernel_u,
    energy_kernel_g,
)
from ad_afqmc_prototype.trial.uhf import UhfTrial
#from ad_afqmc import wavefunctions

def _rand_orthonormal(key: jax.Array, n: int, k: int) -> jax.Array:
    a = jax.random.normal(key, (n, k)) + 1.0j * jax.random.normal(key, (n, k))
    q, _ = jnp.linalg.qr(a)
    return q[:, :k]

#def prep_old(mo_coeff, h0, h1, chol):
#    norb = mo_coeff[0].shape[0]
#    nocc = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
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
#    ham_data["rot_h1"] = [
#        wave_data["mo_coeff"][0].T.conj() @ ham_data["h1"][0],
#        wave_data["mo_coeff"][1].T.conj() @ ham_data["h1"][1],
#    ]
#    ham_data["rot_chol"] = [
#        jnp.einsum(
#            "pi,gij->gpj",
#            wave_data["mo_coeff"][0].T.conj(),
#            ham_data["chol"].reshape(-1, norb, norb),
#        ),
#        jnp.einsum(
#            "pi,gij->gpj",
#            wave_data["mo_coeff"][1].T.conj(),
#            ham_data["chol"].reshape(-1, norb, norb),
#        ),
#    ]
#
#    trial = wavefunctions.uhf(norb=norb, nelec=nocc)
#
#    return trial, ham_data, wave_data

def test_uhf_against_ref():
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    norb=5
    nocc = (2, 3)
    n_chol = 4

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

    ca = _rand_orthonormal(k3, norb, nocc[0]).astype(jnp.complex128)
    cb = _rand_orthonormal(k4, norb, nocc[1]).astype(jnp.complex128)
    cg = _rand_orthonormal(k3, 2*norb, nocc[0]+nocc[1]).astype(jnp.complex128)
    tr = UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)
    ctx = build_meas_ctx(ham, tr)

    wa = jax.random.normal(k1, shape=(norb, nocc[0]))
    wb = jax.random.normal(k2, shape=(norb, nocc[1]))
    wg = jax.random.normal(k1, shape=(2*norb, nocc[0]+nocc[1]))

    fb_u_new = force_bias_kernel_u((wa, wb), None, ctx, tr)
    e_u_new = energy_kernel_u((wa, wb), ham, ctx, tr)
    e_g_new = energy_kernel_g(wg, ham, ctx, tr)

    #trial, ham_data, wave_data = prep_old((ca, cb), h0, h1, chol)
    #fb_u_old = trial._calc_force_bias_unrestricted(wa, wb, ham_data, wave_data)
    #e_u_old = trial._calc_energy_unrestricted(wa, wb, ham_data, wave_data)
    #e_g_old = trial._calc_energy_generalized(wg, ham_data, wave_data)

    fb_u_old = jnp.array([
        5.19102478e-03+2.07733136e-16j,
        5.30495459e+00-1.53306187e-16j,
        4.76844350e+00-5.19549681e-16j,
        2.32073554e+00+6.76542156e-17j
    ])
    e_u_old = -30.542058323121402-5.190973492631015e-15j
    e_g_old = 7.550054074481496-9.358055553355062e-15j

    assert e_u_new - e_u_old < 1e-12
    assert jnp.max(jnp.abs(fb_u_new - fb_u_old)) < 1e-8
    assert e_g_new - e_g_old < 1e-12

test_uhf_against_ref()
