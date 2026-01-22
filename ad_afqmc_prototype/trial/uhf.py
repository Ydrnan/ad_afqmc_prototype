from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UhfTrial:
    mo_coeff_a: jax.Array  # (norb, nocc[0])
    mo_coeff_b: jax.Array  # (norb, nocc[1])

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_a.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (
            int(self.mo_coeff_a.shape[1]),
            int(self.mo_coeff_b.shape[1])
        )

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (mo_coeff_a, mo_coeff_b) = children
        return cls(
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b
        )

def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)

def get_rdm1(trial_data: UhfTrial) -> jax.Array:
    c_a = trial_data.mo_coeff_a
    c_b = trial_data.mo_coeff_b
    dm_a = c_a @ c_a.conj().T  # (norb, norb)
    dm_b = c_b @ c_b.conj().T  # (norb, norb)
    return jnp.stack([dm_a, dm_b], axis=0)  # (2, norb, norb)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UhfTrial) -> jax.Array:
    wu, wd = walker
    cu = trial_data.mo_coeff_a.conj().T @ wu  # (nocc_a, nocc_a)
    cd = trial_data.mo_coeff_b.conj().T @ wd  # (nocc_b, nocc_b)
    return _det(cu) * _det(cd)


#def overlap_g(walker: jax.Array, trial_data: UhfTrial) -> jax.Array:
#    norb = trial_data.norb
#    cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
#    top = cH @ walker[:norb, :]  # (nocc, 2*nocc)
#    bot = cH @ walker[norb:, :]  # (nocc, 2*nocc)
#    m = jnp.vstack([top, bot])  # (2*nocc, 2*nocc)
#    return _det(m)


def make_uhf_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        raise ValueError(f"Cannot use {sys.walker_kind} walker with UHF.")

    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        raise NotImplementedError
        return TrialOps(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
