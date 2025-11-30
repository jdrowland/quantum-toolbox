"""Diagonalization circuit generation for Pauli groups."""

from typing import List, Tuple
import numpy as np
import cirq

from qtoolbox.core.pauli import PauliString
from qtoolbox.converters.cirq_bridge import to_cirq


def diagonalize_pauli_group(
    paulis: List[PauliString],
    qubits: List[cirq.Qid]
) -> Tuple[cirq.Circuit, List[PauliString]]:
    """Generate circuit to diagonalize a group of commuting Paulis."""
    n_qubits = len(qubits)

    # Convert to Cirq for diagonalization
    cirq_paulis = [to_cirq(p, qubits) for p in paulis]

    # Build stabilizer matrix
    stabilizer_matrix = np.zeros((len(paulis), 2 * n_qubits), dtype=int)
    for i, cirq_p in enumerate(cirq_paulis):
        for qubit, pauli in cirq_p.items():
            qubit_idx = qubits.index(qubit)
            if pauli == cirq.X:
                stabilizer_matrix[i, qubit_idx] = 1
            elif pauli == cirq.Y:
                stabilizer_matrix[i, qubit_idx] = 1
                stabilizer_matrix[i, qubit_idx + n_qubits] = 1
            elif pauli == cirq.Z:
                stabilizer_matrix[i, qubit_idx + n_qubits] = 1

    # CRITICAL: Extract linearly independent rows before passing to circuit generation
    # When a group has more Paulis than qubits (e.g., 80 Paulis on 14 qubits),
    # many are linearly dependent. The brute-force search needs rank = n_independent,
    # not rank = n_paulis. This is the key step repacking uses!
    reduced_stabilizer_matrix = _get_linearly_independent_rows(stabilizer_matrix)

    # Gaussian elimination to get measurement circuit (only for independent Paulis)
    circuit, diag_matrix = get_measurement_circuit_from_stabilizers(
        reduced_stabilizer_matrix, qubits
    )

    # Conjugate ALL original Paulis by the circuit (including dependent ones!)
    # CRITICAL: Clifford conjugation can introduce sign flips (±1) and phases (±i).
    # We need to preserve the SIGN but keep the original magnitude.
    # Extract: sign/phase from conjugation result, magnitude from original
    diag_paulis = []
    for orig_pauli, cirq_p in zip(paulis, cirq_paulis):
        original_coeff = orig_pauli.coeff
        diag_p = conjugate_by_circuit(cirq_p, circuit, qubits)
        # Convert back to symplectic
        from qtoolbox.converters.cirq_bridge import from_cirq
        diag_pauli_symplectic = from_cirq(diag_p, n_qubits)

        # Extract the phase/sign from conjugation: phase = conjugated_coeff / original_coeff
        # Then apply: final_coeff = original_coeff * phase
        # But since conjugated already has original_coeff * phase, we can use it directly
        # UNLESS there's magnitude change (which shouldn't happen for Clifford gates)
        conjugated_coeff = diag_p.coefficient
        if abs(original_coeff) > 1e-15:
            phase = conjugated_coeff / original_coeff
            # Apply the phase to original coefficient
            diag_pauli_symplectic.coeff = original_coeff * phase
        else:
            diag_pauli_symplectic.coeff = original_coeff

        diag_paulis.append(diag_pauli_symplectic)

    return circuit, diag_paulis


def _get_linearly_independent_rows(stabilizer_matrix: np.ndarray) -> np.ndarray:
    """Extract linearly independent rows from stabilizer matrix."""
    bool_matrix = stabilizer_matrix.astype(bool)
    # Transpose to make columns=Paulis, then do Gaussian elimination
    reduced_matrix = _binary_gaussian_elimination(bool_matrix.T)

    # Find pivot columns (which are Paulis in the transposed matrix)
    next_pivot = 0
    pivot_columns: List[int] = []

    for j in range(reduced_matrix.shape[1]):
        if next_pivot >= reduced_matrix.shape[0]:
            break
        if reduced_matrix[next_pivot, j]:
            pivot_columns.append(j)
            next_pivot += 1

    # Extract the independent Paulis (rows in original matrix)
    independent_rows = stabilizer_matrix[pivot_columns, :]
    return independent_rows


def get_measurement_circuit_from_stabilizers(
    stabilizer_matrix: np.ndarray,
    qubits: List[cirq.Qid]
) -> Tuple[cirq.Circuit, np.ndarray]:
    """Generate measurement circuit from stabilizer matrix."""
    from itertools import product

    n_qubits = len(qubits)
    n_paulis = stabilizer_matrix.shape[0]

    # Convert our (n_paulis, 2*n_qubits) format to repacking's (2*n_qubits, n_paulis) format
    # Our format: rows are paulis, first half columns are X, second half are Z
    # Repacking format: rows are qubits, first half rows are Z, second half are X
    # So we need: repacking_matrix[i, j] = our_matrix[j, i] for Z part
    #             repacking_matrix[i + n_qubits, j] = our_matrix[j, i + n_qubits] for X part
    z_matrix = stabilizer_matrix[:, n_qubits:].T  # Shape: (n_qubits, n_paulis)
    x_matrix = stabilizer_matrix[:, :n_qubits].T  # Shape: (n_qubits, n_paulis)

    circuit = cirq.Circuit()

    # Find combination of rows to make X matrix have rank n_paulis
    # This tries all 2^n_qubits combinations of choosing Z or X row for each qubit
    for row_combination in product(['X', 'Z'], repeat=n_qubits):
        candidate_matrix = np.array([
            z_matrix[i] if c == "Z" else x_matrix[i]
            for i, c in enumerate(row_combination)
        ])

        # Check if this gives us full rank
        rank = _binary_matrix_rank(candidate_matrix.astype(bool))
        if rank == n_paulis:
            # Apply Hadamards where we chose Z rows
            for i, c in enumerate(row_combination):
                if c == "Z":
                    z_matrix[i] = x_matrix[i].copy()
                    circuit.append(cirq.H(qubits[i]))
            x_matrix = candidate_matrix.copy()
            break

    # Forward elimination
    for j in range(min(n_paulis, n_qubits)):
        # Find pivot
        if x_matrix[j, j] == 0:
            found = False
            for i in range(j + 1, n_qubits):
                if x_matrix[i, j] != 0:
                    found = True
                    break

            if found:
                # Swap rows
                x_matrix[[i, j], :] = x_matrix[[j, i], :]
                z_matrix[[i, j], :] = z_matrix[[j, i], :]
                circuit.append(cirq.SWAP(qubits[j], qubits[i]))

        # Eliminate below diagonal
        for i in range(j + 1, n_qubits):
            if x_matrix[i, j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2
                circuit.append(cirq.CNOT(qubits[j], qubits[i]))

    # Backward elimination
    for j in range(min(n_paulis, n_qubits) - 1, 0, -1):
        for i in range(j):
            if x_matrix[i, j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2
                circuit.append(cirq.CNOT(qubits[j], qubits[i]))

    # Eliminate Z matrix
    for i in range(min(n_paulis, n_qubits)):
        if z_matrix[i, i] == 1:
            for p in range(n_paulis):
                z_matrix[i, p] = (z_matrix[i, p] + x_matrix[i, p]) % 2
            circuit.append(cirq.S(qubits[i]))

        for j in range(i):
            if z_matrix[i, j] == 1:
                for p in range(n_paulis):
                    z_matrix[i, p] = (z_matrix[i, p] + x_matrix[j, p]) % 2
                    z_matrix[j, p] = (z_matrix[j, p] + x_matrix[i, p]) % 2
                circuit.append(cirq.CZ(qubits[j], qubits[i]))

    # Final Hadamards
    for i in range(min(n_paulis, n_qubits)):
        # Swap X and Z rows
        x_matrix[i], z_matrix[i] = z_matrix[i].copy(), x_matrix[i].copy()
        circuit.append(cirq.H(qubits[i]))

    # Convert back to our format (n_paulis, 2*n_qubits)
    # Our X part is transpose of x_matrix, our Z part is transpose of z_matrix
    final_matrix = np.zeros((n_paulis, 2 * n_qubits), dtype=int)
    final_matrix[:, :n_qubits] = x_matrix.T
    final_matrix[:, n_qubits:] = z_matrix.T

    return circuit, final_matrix


def _binary_matrix_rank(mat: np.ndarray) -> int:
    """Compute rank of binary matrix using Gaussian elimination."""
    mat_reduced = _binary_gaussian_elimination(mat)
    num_pivots = 0
    next_pivot = 0

    for j in range(mat_reduced.shape[1]):
        if next_pivot >= mat_reduced.shape[0]:
            break
        if next_pivot < mat_reduced.shape[0] - 1:
            all_zero_below = np.all(np.invert(mat_reduced[(next_pivot + 1):, j]))
        else:
            all_zero_below = True
        if mat_reduced[next_pivot, j] and all_zero_below:
            num_pivots += 1
            next_pivot += 1

    return num_pivots


def _binary_gaussian_elimination(matrix: np.ndarray) -> np.ndarray:
    """Do Gaussian elimination on binary matrix to get RREF."""
    next_row = 0
    mat = matrix.copy()

    for j in range(mat.shape[1]):
        found = False
        for i in range(next_row, mat.shape[0]):
            if mat[i, j]:
                found = True
                if i != next_row:
                    mat[[next_row, i], :] = mat[[i, next_row], :]
                break

        if found:
            for i in range(next_row + 1, mat.shape[0]):
                if mat[i, j]:
                    mat[i, :] ^= mat[next_row, :]
            next_row += 1

    return mat


def conjugate_by_circuit(
    pauli: cirq.PauliString,
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid]
) -> cirq.PauliString:
    """Conjugate a Pauli string by a Clifford circuit."""
    # Use Cirq's built-in conjugation
    result = pauli
    for moment in circuit:
        for op in moment:
            result = result.after(op)
    return result
