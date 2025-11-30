"""DMRG ground state optimization using quimb."""

import sys
import pickle
import logging
from typing import Optional, Union, Tuple
from pathlib import Path

import numpy as np
import cirq
import openfermion as of
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

sys.path.insert(0, '/mnt/ffs24/home/rowlan91/kcommute2')
from kcommute.tensor_nets import pauli_string_to_mpo

from qtoolbox.core.group import GroupCollection
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.converters.openfermion_bridge import to_openfermion

logger = logging.getLogger(__name__)

_PAULI_MATS = {
    (0, 0): np.eye(2, dtype=complex),
    (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
    (0, 1): np.array([[1, 0], [0, -1]], dtype=complex),
    (1, 1): np.array([[0, -1j], [1j, 0]], dtype=complex)
}


def pauli_to_mpo_direct(pauli, n_qubits: int) -> MatrixProductOperator:
    """Build MPO directly from symplectic PauliString."""
    tensors = []
    for i in range(n_qubits):
        x_bit = (pauli.x_bits >> i) & 1
        z_bit = (pauli.z_bits >> i) & 1
        mat = _PAULI_MATS[(x_bit, z_bit)]

        if i == 0:
            tensors.append(mat.reshape(2, 2, 1))
        elif i == n_qubits - 1:
            tensors.append(mat.reshape(1, 2, 2))
        else:
            tensors.append(mat.reshape(1, 2, 2, 1))

    return pauli.coeff * MatrixProductOperator(tensors, shape="ludr")


def build_hamiltonian_mpo(
    source: Union[GroupCollection, Hamiltonian],
    n_qubits: int,
    max_bond: int = 100,
    compress_every: int = 100,
    verbose: bool = False
) -> MatrixProductOperator:
    """Build Hamiltonian as MPO via OpenFermion conversion."""
    qubits = cirq.LineQubit.range(n_qubits)

    if isinstance(source, GroupCollection):
        paulis = [p for g in source.groups for p in g.paulis]
    elif isinstance(source, Hamiltonian):
        paulis = source.terms
    else:
        raise TypeError(f"Expected GroupCollection or Hamiltonian, got {type(source)}")

    if verbose:
        print(f"Building Hamiltonian MPO from {len(paulis)} terms...", flush=True)

    mpo = None
    for i, pauli in enumerate(paulis):
        qubop = to_openfermion(pauli)
        psum = of.transforms.qubit_operator_to_pauli_sum(qubop)

        for pstring in psum:
            term_mpo = pauli_string_to_mpo(pstring, qubits)

            if mpo is None:
                mpo = term_mpo
            else:
                mpo = mpo + term_mpo

                if (i + 1) % compress_every == 0:
                    tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)

        if verbose and (i + 1) % 500 == 0:
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
            print(f"  {i+1}/{len(paulis)} terms, bond dims: {mpo.bond_sizes()}", flush=True)

    tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)

    if verbose:
        print(f"Final MPO bond dims: {mpo.bond_sizes()}", flush=True)

    return mpo


def build_hamiltonian_mpo_direct(
    source: Union[GroupCollection, Hamiltonian],
    n_qubits: int,
    max_bond: int = 100,
    compress_every: int = 100,
    verbose: bool = False
) -> MatrixProductOperator:
    """Build Hamiltonian MPO using direct symplectic->MPO conversion."""
    if isinstance(source, GroupCollection):
        paulis = [p for g in source.groups for p in g.paulis]
    elif isinstance(source, Hamiltonian):
        paulis = source.terms
    else:
        raise TypeError(f"Expected GroupCollection or Hamiltonian, got {type(source)}")

    if verbose:
        print(f"Building Hamiltonian MPO from {len(paulis)} terms (direct)...", flush=True)

    mpo = None
    for i, pauli in enumerate(paulis):
        term_mpo = pauli_to_mpo_direct(pauli, n_qubits)

        if mpo is None:
            mpo = term_mpo
        else:
            mpo = mpo + term_mpo

            if (i + 1) % compress_every == 0:
                tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)

        if verbose and (i + 1) % 500 == 0:
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
            print(f"  {i+1}/{len(paulis)} terms, bond dims: {mpo.bond_sizes()}", flush=True)

    tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)

    if verbose:
        print(f"Final MPO bond dims: {mpo.bond_sizes()}", flush=True)

    return mpo


def run_dmrg(
    hamiltonian_mpo: MatrixProductOperator,
    max_bond_dim: int,
    n_sweeps: int = 20,
    tol: float = 1e-6,
    initial_state: Optional[MatrixProductState] = None,
    verbose: bool = False
) -> Tuple[MatrixProductState, float]:
    """Run DMRG to find ground state MPS."""
    verbosity = 1 if verbose else 0

    dmrg = qtn.DMRG2(
        hamiltonian_mpo,
        which='SA',
        bond_dims=[max_bond_dim],
        p0=initial_state
    )

    dmrg.solve(max_sweeps=n_sweeps, tol=tol, verbosity=verbosity)

    energy = dmrg.energy.real if np.iscomplex(dmrg.energy) else dmrg.energy
    state = dmrg.state

    if verbose:
        print(f"DMRG converged: E = {energy:.8f} Ha")
        print(f"MPS bond dims: {state.bond_sizes()}")

    return state, energy


def create_hf_mps(n_qubits: int, n_electrons: int) -> MatrixProductState:
    """Create Hartree-Fock reference state as MPS."""
    arrays = []
    for i in range(n_qubits):
        if i < n_electrons:
            arrays.append(np.array([0.0, 1.0]))
        else:
            arrays.append(np.array([1.0, 0.0]))

    mps = MatrixProductState(
        arrays=[a.reshape(1, 2, 1) for a in arrays],
        shape='lpr'
    )
    return mps


def save_mps(mps: MatrixProductState, filepath: Union[str, Path]) -> None:
    """Save MPS state to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(mps, f)


def load_mps(filepath: Union[str, Path]) -> MatrixProductState:
    """Load MPS state from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compute_energy_from_mps(
    mps: MatrixProductState,
    hamiltonian_mpo: MatrixProductOperator
) -> float:
    """Compute energy expectation value <ψ|H|ψ>."""
    h_psi = hamiltonian_mpo.apply(mps)
    energy = mps.H @ h_psi
    return energy.real if np.iscomplex(energy) else energy
