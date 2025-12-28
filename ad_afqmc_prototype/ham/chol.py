from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
from jax import tree_util

ham_basis = Literal["restricted", "generalized"]


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamChol:
    """
    cholesky hamiltonian.

    basis="restricted":
      h1:   (norb, norb)
      chol: (n_fields, norb, norb)

    basis="generalized":
      h1:   (nso, nso)   where nso = 2*norb
      chol: (n_fields, nso, nso)
    """

    h0: jax.Array
    h1: jax.Array
    chol: jax.Array
    basis: ham_basis = "restricted"

    def __post_init__(self):
        if self.basis not in ("restricted", "generalized"):
            raise ValueError(f"unknown basis: {self.basis}")

    def tree_flatten(self):
        children = (self.h0, self.h1, self.chol)
        aux = self.basis
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        h0, h1, chol = children
        basis = aux
        return cls(h0=h0, h1=h1, chol=chol, basis=basis)


def n_fields(ham: HamChol) -> int:
    return int(ham.chol.shape[0])
