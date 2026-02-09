from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from .ham.chol import HamBasis

import h5py
import numpy as np

# This file contains staging utilities to convert pyscf mf/cc objects
# into serializable data classes representing Hamiltonian and trial
# wavefunction inputs which can be used for building AFQMC objects.

Array = np.ndarray

# to keep track of format versions when loading/saving staged inputs
STAGE_FORMAT_VERSION = 1


def modified_cholesky(
    mat: Array,
    max_error: float = 1e-6,
) -> Array:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (Array): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        Array: Cholesky vectors.
    """
    diag = mat.diagonal()
    norb = int(((-1 + (1 + 8 * mat.shape[0]) ** 0.5) / 2))
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    chol0 = chol_vecs[:nchol]
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
        for m in range(norb):
            for n in range(m + 1):
                triind = m * (m + 1) // 2 + n
                chol[i, m, n] = chol0[i, triind]
                chol[i, n, m] = chol0[i, triind]
    return chol


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems. (copied from pauxy)

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


@dataclass(frozen=True, slots=True)
class HamInput:
    """ham inputs in the chosen orthonormal one particle basis"""

    h0: float
    h1: Array  # (norb, norb)
    chol: Array  # (nchol, norb, norb)
    nelec: Tuple[int, int]
    norb: int
    chol_cut: float
    norb_frozen: int
    source_kind: str  # "mf" or "cc"
    basis: HamBasis # "restricted" or "generalized"


@dataclass(frozen=True, slots=True)
class TrialInput:
    """trial inputs used to construct an afqmc trial object"""

    kind: str  # "slater", "cisd", "ucisd"
    data: Dict[str, Array]
    norb_frozen: int
    source_kind: str  # "mf" or "cc"


@dataclass(frozen=True, slots=True)
class StagedInputs:
    ham: HamInput
    trial: TrialInput
    meta: Dict[str, Any]


# public API
def stage(
    obj: Any,
    *,
    norb_frozen: int = 0,
    chol_cut: float = 1e-5,
    cache: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> StagedInputs:
    """
    Stage inputs from a pyscf mf or cc object.

    Args:
        obj:
            pyscf mf object (RHF/ROHF/UHF) or cc object (CCSD/UCCSD).
        norb_frozen:
            Number of lowest orbitals to freeze.
            Inferred from cc.frozen if obj is a cc object.
        chol_cut:
            Cholesky decomposition cutoff.
        cache:
            Optional path to write on disk. If it exists and overwrite=False,
            loads it. Otherwise computes and writes it.
        overwrite:
            If True and cache is provided, recompute and overwrite cache.
        verbose:
            Print timing/info.

    Returns:
        StagedInputs containing HamInput, TrialInput, and metadata.
    """
    cache_path = Path(cache).expanduser().resolve() if cache is not None else None
    if cache_path is not None and cache_path.exists() and not overwrite:
        return load(cache_path)

    t0 = time.time()
    scf_obj, source_kind = _normalize_source(obj)
    mol = scf_obj.mol

    # TODO: Replace the check. Since norb_frozen is override here,
    # the later call to _check_cc_frozen_consistency has no effect.
    # Also the condistency check must be done before computing the
    # integrals, not after.
    if source_kind == "cc":
        if isinstance(obj.frozen, int):
            norb_frozen = obj.frozen

    ham = _stage_ham_input(
        scf_obj,
        source_kind=source_kind,
        norb_frozen=norb_frozen,
        chol_cut=chol_cut,
        verbose=verbose,
    )

    trial = _stage_trial_input(
        obj, scf_obj=scf_obj, source_kind=source_kind, norb_frozen=norb_frozen
    )

    meta: Dict[str, Any] = {
        "format_version": STAGE_FORMAT_VERSION,
        "timestamp_unix": time.time(),
        "source_kind": source_kind,
        "norb_frozen": norb_frozen,
        "chol_cut": chol_cut,
        "mol": {
            "nao": int(mol.nao),
            "nelectron": int(mol.nelectron),
            "spin": int(mol.spin),
            "charge": int(mol.charge),
            "basis": getattr(mol, "basis", None),
        },
    }

    staged = StagedInputs(ham=ham, trial=trial, meta=meta)

    if cache_path is not None:
        dump(staged, cache_path)

    if verbose:
        dt = time.time() - t0
        print(f"[stage] done in {dt:.2f}s | norb={ham.norb} nchol={ham.chol.shape[0]}")

    return staged


def dump(staged: StagedInputs, path: Union[str, Path]) -> None:
    """
    Save staged inputs to a single h5 file.

    Args:
        staged: StagedInputs to serialize
        path: output file path
    """
    p = Path(path).expanduser().resolve()
    _dump_h5(staged, p)


def load(path: Union[str, Path]) -> StagedInputs:
    """
    Load staged inputs from a single file written by dump().

    Args:
        path: input file path

    Returns:
        StagedInputs
    """
    p = Path(path).expanduser().resolve()
    return _load_h5(p)


def _is_cc_like(obj: Any) -> bool:
    return hasattr(obj, "t1") and hasattr(obj, "t2")

def _is_mf_like(obj: Any) -> bool:
    return hasattr(obj, "mo_coeff")

def _normalize_source(obj: Any) -> tuple[Any, str]:
    """
    Return (mf_obj, kind) where kind is "mf" or "cc".
    """
    if _is_cc_like(obj):
        mf = getattr(obj, "_scf", None)
        if mf is None:
            raise TypeError(
                "CC-like object missing _scf reference to underlying scf object."
            )
        return mf, "cc"
    return obj, "mf"


def _stage_ham_input(
    scf_obj: Any, *, source_kind: str, norb_frozen: int, chol_cut: float, verbose: bool
) -> HamInput:
    """
    Produce h0/h1/chol in a single orthonormal basis.
    For UHF, we use the alpha MO basis for integrals.
    """
    from pyscf import mcscf
    from pyscf.scf.ghf import GHF
    from pyscf.scf.hf import RHF
    from pyscf.scf.rohf import ROHF
    from pyscf.scf.uhf import UHF

    mol = scf_obj.mol

    if isinstance(scf_obj, (RHF, ROHF, GHF)):
        basis_coeff = np.asarray(scf_obj.mo_coeff)
    elif isinstance(scf_obj, UHF):
        basis_coeff = np.asarray(scf_obj.mo_coeff[0])
    else:
        raise TypeError(f"Unsupported SCF type for staging: {type(scf_obj)}")

    if isinstance(scf_obj, (RHF, ROHF, UHF)):
        ham_basis = "restricted"
    elif isinstance(scf_obj, GHF):
        ham_basis = "generalized"
    else:
        raise TypeError(f"Unreachable")

    # nuclear energy (without frozen core correction)
    h0 = float(scf_obj.energy_nuc())

    # one body
    hcore = scf_obj.get_hcore()
    h1 = basis_coeff.T.conj() @ hcore @ basis_coeff
    h1 = np.asarray(h1)

    # ao cholesky
    t0 = time.time()
    chol_vec = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print(
            f"[stage] AO cholesky: nchol={chol_vec.shape[0]} in {time.time() - t0:.2f}s"
        )

    # full space electron count
    nelec: Tuple[int, int] = (int(mol.nelec[0]), int(mol.nelec[1]))

    # mo Cholesky
    nchol = int(chol_vec.shape[0])
    C = np.asarray(basis_coeff)
    if not isinstance(scf_obj, GHF):
        norb = int(basis_coeff.shape[1])
        Cdag = C.conj().T
        chol_ao = chol_vec.reshape(nchol, mol.nao, mol.nao)
        tmp = Cdag @ chol_ao
        chol = tmp @ C
    else:
        import scipy.linalg as la
        norb = basis_coeff.shape[1] // 2
        chol = np.zeros((nchol, 2*norb, 2*norb), dtype=C.dtype)
        for i in range(nchol):
            chol_i = chol_vec[i].reshape(norb, norb)
            bchol_i = la.block_diag(chol_i, chol_i)
            chol[i] = C.T.conj() @ bchol_i @ C

    # freeze core
    if norb_frozen > 0 and not isinstance(scf_obj, GHF):
        if norb_frozen > min(nelec):
            raise ValueError(
                f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}"
            )

        nelec_frozen = 2 * norb_frozen
        ncas = basis_coeff.shape[1] - norb_frozen
        nelecas = mol.nelectron - nelec_frozen
        if nelecas <= 0 or ncas <= 0:
            raise ValueError("Frozen core left no active electrons/orbitals.")

        mc = mcscf.CASSCF(scf_obj, ncas, nelecas)
        mc.mo_coeff = basis_coeff  # type: ignore
        h1_eff, ecore = mc.get_h1eff()  # type: ignore
        i0 = int(mc.ncore)  # type: ignore
        i1 = i0 + int(mc.ncas)  # type: ignore

        h0 = float(ecore)
        h1 = np.asarray(h1_eff)
        chol = chol[:, i0:i1, i0:i1]
        norb = int(ncas)
        nelec = tuple(int(x) for x in mc.nelecas)  # type: ignore
    elif norb_frozen > 0 and isinstance(scf_obj, GHF):
        raise NotImplementedError(f"Frozen core approximation not available for generalised integrals.")

    return HamInput(
        h0=h0,
        h1=np.asarray(h1),
        chol=np.asarray(chol),
        nelec=nelec,
        norb=norb,
        chol_cut=float(chol_cut),
        norb_frozen=int(norb_frozen),
        source_kind=source_kind,
        basis=ham_basis,
    )

def _stage_trial_input(
    obj: Any, *, scf_obj: Any, source_kind: str, norb_frozen: int
) -> TrialInput:
    """
    Produce TrialInput consistent with the Hamiltonian basis and frozen core choice
    """
    from pyscf.scf.hf import RHF
    from pyscf.scf.rohf import ROHF
    from pyscf.scf.uhf import UHF
    from pyscf.scf.ghf import GHF
    from pyscf.cc.ccsd import CCSD
    from pyscf.cc.uccsd import UCCSD
    from pyscf.cc.gccsd import GCCSD

    if _is_cc_like(obj):
        if isinstance(obj, CCSD):
            stage_tr_fun = _stage_cisd_input
        elif isinstance(obj, UCCSD):
            stage_tr_fun = _stage_ucisd_input
        elif isinstance(obj, GCCSD):
            stage_tr_fun = _stage_gcisd_input
        else:
            raise TypeError(f"Unsupported CC type: {type(obj)}")
    elif _is_mf_like(obj):
        obj = scf_obj
        if isinstance(obj, (RHF, ROHF, UHF, GHF)):
            stage_tr_fun = _stage_mf_input
        else:
            raise TypeError(f"Unsupported SCF type: {type(obj)}")
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    return stage_tr_fun(
        obj,
        source_kind=source_kind,
        norb_frozen=norb_frozen,
    )


def _stage_mf_input(
    scf_obj: Any,
    *,
    source_kind: str,
    norb_frozen: int,
) -> TrialInput:
    from pyscf.scf.hf import RHF
    from pyscf.scf.rohf import ROHF
    from pyscf.scf.uhf import UHF
    from pyscf.scf.ghf import GHF

    assert isinstance(
        scf_obj, (RHF, ROHF, UHF, GHF)
    ), f"Unsupported obj type: {type(scf_obj)}"

    mol = scf_obj.mol
    S = scf_obj.get_ovlp(mol)

    if isinstance(scf_obj, (RHF, ROHF, GHF)):
        Ca = np.asarray(scf_obj.mo_coeff)
        mo = _mf_coeff_helper(Ca, Ca, S, norb_frozen)
        data = {"mo": np.asarray(mo)}

    if isinstance(scf_obj, UHF):
        Ca = np.asarray(scf_obj.mo_coeff[0])
        Cb = np.asarray(scf_obj.mo_coeff[1])

        # basis is alpha MOs, represent alpha and beta orbitals in this basis
        moa = _mf_coeff_helper(Ca, Ca, S, norb_frozen)
        mob = _mf_coeff_helper(Ca, Cb, S, norb_frozen)
        data = {"mo_a": np.asarray(moa), "mo_b": np.asarray(mob)}

    if isinstance(scf_obj, RHF):
        kind = "rhf"
    elif isinstance(scf_obj, ROHF) or isinstance(scf_obj, UHF):
        kind = "uhf"
    elif isinstance(scf_obj, GHF):
        kind = "ghf"
    else:
        raise TypeError(f"Unreachable")

    return TrialInput(
        kind=kind,
        data=data,
        norb_frozen=int(norb_frozen),
        source_kind=source_kind,
    )


def _mf_coeff_helper(
    Ca: np.Array,
    Cb: np.Array,
    S: np.Array,
    norb_frozen: int,
) -> np.Array:
    q, r = np.linalg.qr(Ca.T @ S @ Cb)
    sgn = np.sign(np.diag(r))
    q = q * sgn[None, :]
    q = q[norb_frozen:, norb_frozen:]

    return q


def _stage_cisd_input(
    cc_obj: Any,
    *,
    source_kind: str,
    norb_frozen: int,
) -> TrialInput:
    from pyscf.cc.ccsd import CCSD

    if not isinstance(cc_obj, CCSD):
        raise TypeError(f"ansatz='cisd' expects a CCSD object, got: {type(cc_obj)}")
    _check_cc_frozen_consistency(cc_obj, norb_frozen)

    if not hasattr(cc_obj, "t1") or not hasattr(cc_obj, "t2"):
        raise ValueError("CC amplitudes not found; did you run cc.kernel()?")

    ci2 = np.asarray(cc_obj.t2) + np.einsum(
        "ia,jb->ijab", np.asarray(cc_obj.t1), np.asarray(cc_obj.t1)
    )
    ci2 = ci2.transpose(0, 2, 1, 3)  # (i,a,j,b) -> (i,j,a,b)
    ci1 = np.asarray(cc_obj.t1)

    data = {"ci1": ci1, "ci2": ci2}
    return TrialInput(
        kind="cisd",
        data=data,
        norb_frozen=int(norb_frozen),
        source_kind=source_kind,
    )


def _stage_ucisd_input(
    cc_obj: Any,
    *,
    source_kind: str,
    norb_frozen: int,
) -> TrialInput:
    from pyscf.cc.uccsd import UCCSD

    if not isinstance(cc_obj, UCCSD):
        raise TypeError(f"ansatz='ucisd' expects a UCCSD object, got: {type(cc_obj)}")
    _check_cc_frozen_consistency(cc_obj, norb_frozen)

    if not hasattr(cc_obj, "t1") or not hasattr(cc_obj, "t2"):
        raise ValueError("CC amplitudes not found; did you run cc.kernel()?")

    t1a, t1b = cc_obj.t1
    t2aa, t2ab, t2bb = cc_obj.t2

    ci2aa = np.asarray(t2aa) + 2.0 * np.einsum(
        "ia,jb->ijab", np.asarray(t1a), np.asarray(t1a)
    )
    ci2aa = 0.5 * (ci2aa - ci2aa.transpose(0, 1, 3, 2))
    ci2aa = ci2aa.transpose(0, 2, 1, 3)

    ci2bb = np.asarray(t2bb) + 2.0 * np.einsum(
        "ia,jb->ijab", np.asarray(t1b), np.asarray(t1b)
    )
    ci2bb = 0.5 * (ci2bb - ci2bb.transpose(0, 1, 3, 2))
    ci2bb = ci2bb.transpose(0, 2, 1, 3)

    ci2ab = np.asarray(t2ab) + np.einsum(
        "ia,jb->ijab", np.asarray(t1a), np.asarray(t1b)
    )
    ci2ab = ci2ab.transpose(0, 2, 1, 3)

    scf_obj = cc_obj._scf
    _uhf_input = _stage_mf_input(
        scf_obj,
        source_kind=source_kind,
        norb_frozen=norb_frozen,
    )
    moa = _uhf_input.data["mo_a"]
    mob = _uhf_input.data["mo_b"]

    data = {
        "mo_coeff_a": np.asarray(moa),
        "mo_coeff_b": np.asarray(mob),
        "ci1a": np.asarray(t1a),
        "ci1b": np.asarray(t1b),
        "ci2aa": np.asarray(ci2aa),
        "ci2ab": np.asarray(ci2ab),
        "ci2bb": np.asarray(ci2bb),
    }
    return TrialInput(
        kind="ucisd",
        data=data,
        norb_frozen=int(norb_frozen),
        source_kind=source_kind,
    )


def _stage_gcisd_input(
    cc_obj: Any,
    *,
    source_kind: str,
    norb_frozen: int,
) -> TrialInput:
    from pyscf.cc.gccsd import GCCSD

    if not isinstance(cc_obj, GCCSD):
        raise TypeError(f"ansatz='gcisd' expects a GCCSD object, got: {type(cc_obj)}")
    _check_cc_frozen_consistency(cc_obj, norb_frozen)

    if not hasattr(cc_obj, "t1") or not hasattr(cc_obj, "t2"):
        raise ValueError("CC amplitudes not found; did you run cc.kernel()?")

    ci2 = (
        np.einsum("ijab->iajb", cc_obj.t2)
        + np.einsum("ia,jb->iajb", cc_obj.t1, cc_obj.t1)
        - np.einsum("ib,ja->iajb", cc_obj.t1, cc_obj.t1)
    )
    ci1 = np.asarray(cc_obj.t1)

    scf_obj = cc_obj._scf
    _ghf_input = _stage_mf_input(
        scf_obj,
        source_kind=source_kind,
        norb_frozen=norb_frozen,
    )
    mo = _ghf_input.data["mo"]

    data = {"mo_coeff": mo, "ci1": ci1, "ci2": ci2}
    return TrialInput(
        kind="gcisd",
        data=data,
        norb_frozen=int(norb_frozen),
        source_kind=source_kind,
    )


def _check_cc_frozen_consistency(cc_obj: Any, norb_frozen: int) -> None:
    """
    Assert the user froze orbitals in CC the same as in staging.
    """
    if not hasattr(cc_obj, "frozen") or cc_obj.frozen is None:
        assert (
            norb_frozen == 0
        ), "cc_obj has no frozen attribute, staging frozen must be 0."
    if isinstance(cc_obj.frozen, int):
        cc_frozen = int(cc_obj.frozen)
        assert (
            cc_frozen == norb_frozen
        ), "cc_obj.frozen and staging frozen must be equal."


def _dump_h5(staged: StagedInputs, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(staged.meta)

        gham = f.create_group("ham")
        gham.create_dataset("h0", data=np.array(staged.ham.h0))
        gham.create_dataset("h1", data=staged.ham.h1)
        gham.create_dataset("chol", data=staged.ham.chol)
        gham.create_dataset("nelec", data=np.array(staged.ham.nelec, dtype=np.int64))
        gham.attrs["norb"] = staged.ham.norb
        gham.attrs["chol_cut"] = staged.ham.chol_cut
        gham.attrs["norb_frozen"] = staged.ham.norb_frozen
        gham.attrs["source_kind"] = staged.ham.source_kind
        gham.attrs["basis"] = staged.ham.basis

        gtr = f.create_group("trial")
        gtr.attrs["kind"] = staged.trial.kind
        gtr.attrs["norb_frozen"] = staged.trial.norb_frozen
        gtr.attrs["source_kind"] = staged.trial.source_kind
        gdata = gtr.create_group("data")
        for k, v in staged.trial.data.items():
            gdata.create_dataset(k, data=np.asarray(v))


def _to_json_str(x: Any) -> str:
    # np scalar -> python scalar
    if isinstance(x, np.ndarray):
        x = x.item()
    # bytes like -> decode
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return bytes(x).decode("utf-8")
    return str(x)


def _load_h5(path: Path) -> StagedInputs:
    with h5py.File(path, "r") as f:
        meta = json.loads(_to_json_str(f.attrs["meta_json"]))

        gham: Any = f["ham"]
        ham = HamInput(
            h0=float(np.array(gham["h0"]).item()),
            h1=np.array(gham["h1"]),
            chol=np.array(gham["chol"]),
            nelec=(int(np.array(gham["nelec"])[0]), int(np.array(gham["nelec"])[1])),
            norb=int(gham.attrs["norb"]),
            chol_cut=float(gham.attrs["chol_cut"]),
            norb_frozen=int(gham.attrs["norb_frozen"]),
            source_kind=str(gham.attrs["source_kind"]),
            basis=str(gham.attrs["basis"])
        )

        gtr: Any = f["trial"]
        gdata = gtr["data"]
        trial_data = {k: np.array(gdata[k]) for k in gdata.keys()}
        trial = TrialInput(
            kind=str(gtr.attrs["kind"]),
            data=trial_data,
            norb_frozen=int(gtr.attrs["norb_frozen"]),
            source_kind=str(gtr.attrs["source_kind"]),
        )

        return StagedInputs(ham=ham, trial=trial, meta=meta)
