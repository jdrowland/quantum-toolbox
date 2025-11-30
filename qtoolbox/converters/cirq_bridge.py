"""Conversion between symplectic PauliString and Cirq PauliString."""

from typing import List
import cirq
from qtoolbox.core.pauli import PauliString


def from_cirq(cirq_pauli: cirq.PauliString, n_qubits: int) -> PauliString:
    """Convert Cirq PauliString to symplectic representation."""
    x_bits = 0
    z_bits = 0

    for qubit, pauli in cirq_pauli.items():
        qubit_idx = qubit.x

        if pauli == cirq.X:
            x_bits |= (1 << qubit_idx)
        elif pauli == cirq.Y:
            x_bits |= (1 << qubit_idx)
            z_bits |= (1 << qubit_idx)
        elif pauli == cirq.Z:
            z_bits |= (1 << qubit_idx)

    coeff = complex(cirq_pauli.coefficient)
    return PauliString(x_bits, z_bits, coeff, n_qubits)


def to_cirq(pauli: PauliString, qubits: List[cirq.Qid]) -> cirq.PauliString:
    """Convert symplectic PauliString to Cirq PauliString."""
    if len(qubits) != pauli.n_qubits:
        raise ValueError(f"Qubit list length {len(qubits)} != n_qubits {pauli.n_qubits}")

    pauli_dict = {}
    for i in range(pauli.n_qubits):
        x_bit = (pauli.x_bits >> i) & 1
        z_bit = (pauli.z_bits >> i) & 1

        if x_bit and z_bit:
            pauli_dict[qubits[i]] = cirq.Y
        elif x_bit:
            pauli_dict[qubits[i]] = cirq.X
        elif z_bit:
            pauli_dict[qubits[i]] = cirq.Z

    return cirq.PauliString(pauli_dict, coefficient=pauli.coeff)


def from_cirq_pauli_sum(pauli_sum: cirq.PauliSum, qubits: List[cirq.Qid]) -> List[PauliString]:
    """Convert Cirq PauliSum to list of symplectic PauliStrings."""
    n_qubits = len(qubits)
    return [from_cirq(term, n_qubits) for term in pauli_sum]


def to_cirq_pauli_sum(paulis: List[PauliString], qubits: List[cirq.Qid]) -> cirq.PauliSum:
    """Convert list of symplectic PauliStrings to Cirq PauliSum."""
    return sum(to_cirq(p, qubits) for p in paulis)
