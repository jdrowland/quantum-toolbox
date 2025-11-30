"""Circuit utilities for quantum state preparation."""

import cirq


def get_qubits(n_qubits: int) -> list:
    """Get list of cirq.LineQubits."""
    return cirq.LineQubit.range(n_qubits)
