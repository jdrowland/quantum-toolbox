"""Conversion between symplectic PauliString and OpenFermion QubitOperator."""

from typing import Tuple
import openfermion as of
from qtoolbox.core.pauli import PauliString


def from_openfermion(
    of_term: Tuple[Tuple[int, str], ...],
    coeff: complex,
    n_qubits: int
) -> PauliString:
    """Convert OpenFermion Pauli term to symplectic representation.

    Args:
        of_term: OpenFermion term like ((0, 'X'), (1, 'Y'), (2, 'Z'))
        coeff: Coefficient for this term
        n_qubits: Total number of qubits
    """
    x_bits = 0
    z_bits = 0

    for qubit_idx, pauli in of_term:
        if pauli == 'X':
            x_bits |= (1 << qubit_idx)
        elif pauli == 'Y':
            x_bits |= (1 << qubit_idx)
            z_bits |= (1 << qubit_idx)
        elif pauli == 'Z':
            z_bits |= (1 << qubit_idx)

    return PauliString(x_bits, z_bits, coeff, n_qubits)


def to_openfermion(pauli: PauliString) -> of.QubitOperator:
    """Convert symplectic PauliString to OpenFermion QubitOperator."""
    term = []
    for i in range(pauli.n_qubits):
        x_bit = (pauli.x_bits >> i) & 1
        z_bit = (pauli.z_bits >> i) & 1

        if x_bit and z_bit:
            term.append((i, 'Y'))
        elif x_bit:
            term.append((i, 'X'))
        elif z_bit:
            term.append((i, 'Z'))

    if not term:
        term = ()
    else:
        term = tuple(term)

    return of.QubitOperator(term, pauli.coeff)
