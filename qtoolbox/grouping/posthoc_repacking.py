"""Post-hoc repacking algorithm."""

from typing import Optional, List
import cirq

from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.core.group import GroupCollection, PauliGroup
from qtoolbox.core.pauli import PauliString
from qtoolbox.converters.cirq_bridge import to_cirq


def posthoc_repacking(
    hamiltonian: Hamiltonian,
    baseline_groups: GroupCollection,
    circuits: List[cirq.Circuit],
    qubits: List[cirq.Qid],
    k: Optional[int] = None
) -> GroupCollection:
    """Post-hoc repacking: find Paulis measured by existing baseline circuits."""
    if len(circuits) != baseline_groups.num_groups():
        raise ValueError("Must provide one circuit per group")

    # Create new collection
    repacked = GroupCollection()

    for group_idx, (baseline_group, circuit) in enumerate(zip(baseline_groups.groups, circuits)):
        new_group = PauliGroup()

        # Add baseline Paulis
        for pauli in baseline_group.paulis:
            new_group.add(pauli.copy())

        # Check Paulis from previous groups
        for prev_group_idx in range(group_idx):
            prev_group = baseline_groups.groups[prev_group_idx]

            for pauli in prev_group.paulis:
                # Skip if already in this group
                if pauli in new_group.paulis:
                    continue

                # Check if commutes with all in group
                commutes = True
                if k is None:
                    commutes = all(pauli.commutes_with(p) for p in baseline_group.paulis)
                else:
                    commutes = all(pauli.k_commutes(p, k) for p in baseline_group.paulis)

                if not commutes:
                    continue

                # Check if diagonal under this circuit
                if is_diagonal_under_circuit(pauli, circuit, qubits):
                    new_group.add(pauli.copy())

        repacked.groups.append(new_group)

    return repacked


def is_diagonal_under_circuit(
    pauli: PauliString,
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid]
) -> bool:
    """Check if Pauli is diagonal (only I and Z) after conjugation by circuit."""
    # Convert to Cirq
    cirq_pauli = to_cirq(pauli, qubits)

    # Conjugate by circuit
    conjugated = cirq_pauli
    for moment in circuit:
        for op in moment:
            conjugated = conjugated.after(op)

    # Check if diagonal (only Z and I)
    for qubit, pauli_op in conjugated.items():
        if pauli_op not in [cirq.Z, cirq.I]:
            return False

    return True
