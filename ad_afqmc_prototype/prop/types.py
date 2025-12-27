from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Protocol

import jax

from ..core.ops import meas_ops


class prop_state(NamedTuple):
    walkers: Any
    weights: jax.Array
    overlaps: jax.Array
    rng_key: jax.Array
    pop_control_ene_shift: jax.Array
    e_estimate: jax.Array
    node_encounters: jax.Array


@dataclass(frozen=True)
class afqmc_params:
    dt: float = 0.005
    n_chunks: int = 1
    n_exp_terms: int = 6
    pop_control_damping: float = 0.1
    weight_floor: float = 1.0e-3
    weight_cap: float = 100.0
    n_prop_steps: int = 50
    shift_ema: float = 0.1
    n_eql_blocks: int = 50
    n_blocks: int = 500
    n_walkers: int = 200
    seed: int = 42


class step_kernel(Protocol):

    def __call__(
        self,
        state: prop_state,
        *,
        params: afqmc_params,
        ham_data: Any,
        trial_data: Any,
        meas_ops: meas_ops,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> prop_state: ...


@dataclass(frozen=True)
class prop_ops:
    build_prop_ctx: Callable[
        [Any, Any, afqmc_params], Any
    ]  # (ham_data, trial_data, params) -> prop_ctx
    step: step_kernel
