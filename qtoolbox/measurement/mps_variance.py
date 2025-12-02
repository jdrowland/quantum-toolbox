"""Variance computation from MPS states using MPO-based methods."""

import sys
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
import cirq
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductState

# # Import kcommute2 for MPO-based variance
# sys.path.insert(0, '/mnt/ffs24/home/rowlan91/kcommute2')
# from kcommute.tensor_nets import pauli_sum_to_mpo, mpo_mps_exepctation
from qtoolbox.converters.quimb_bridge import pauli_sum_to_mpo
from qtoolbox.core.tensors import mpo_mps_exepctation

from qtoolbox.core.group import GroupCollection
from qtoolbox.core.pauli import PauliString
from qtoolbox.converters.openfermion_bridge import to_openfermion


@dataclass
class VarianceResult:
    """Results from variance computation."""
    total_variance: float
    sum_sigma_g: float
    shots_for_target: int
    target_stderr: float
    per_group_variance: List[float]
    per_group_sigma: List[float]
    energy: float
    num_terms: int
    num_groups: int


def compute_pauli_expectation(
    pauli: PauliString,
    mps: MatrixProductState,
    qubits: List[cirq.Qid],
    max_bond: int = 50
) -> float:
    """Compute ⟨ψ|P|ψ⟩ for a single Pauli operator."""
    # Convert to cirq PauliString
    qubop = to_openfermion(pauli)
    psum = of.transforms.qubit_operator_to_pauli_sum(qubop)

    # Build MPO for this Pauli
    mpo = pauli_sum_to_mpo(psum, qubits, max_bond)

    # Compute expectation
    expectation = mpo_mps_exepctation(mpo, mps)
    return expectation.real if np.iscomplex(expectation) else expectation


def compute_variance_from_mps(
    groups: GroupCollection,
    mps: MatrixProductState,
    max_bond: int = 50,
    target_stderr: float = 0.003,
    verbose: bool = False
) -> VarianceResult:
    """Compute shot count estimate from MPS state."""
    # Get qubits
    n_qubits = groups.groups[0].paulis[0].n_qubits
    qubits = cirq.LineQubit.range(n_qubits)

    # Track per-Pauli expectations to avoid recomputation
    pauli_expectations: Dict[Tuple[int, int], float] = {}

    per_group_variance = []
    per_group_sigma = []
    total_energy = 0.0
    total_terms = 0

    for g_idx, group in enumerate(groups.groups):
        group_variance = 0.0

        for pauli in group.paulis:
            key = (pauli.x_bits, pauli.z_bits)

            # Get or compute expectation
            if key not in pauli_expectations:
                exp_val = compute_pauli_expectation(pauli, mps, qubits, max_bond)
                pauli_expectations[key] = exp_val
            else:
                exp_val = pauli_expectations[key]

            # Var(P_i) = 1 - ⟨P_i⟩²
            var_p = 1.0 - exp_val ** 2

            # Contribution to group variance: c_i² * Var(P_i)
            coeff = pauli.coeff.real if np.iscomplex(pauli.coeff) else pauli.coeff
            group_variance += coeff ** 2 * var_p

            # Accumulate energy
            total_energy += coeff * exp_val
            total_terms += 1

        sigma_g = np.sqrt(max(0.0, group_variance))
        per_group_variance.append(group_variance)
        per_group_sigma.append(sigma_g)

        if verbose and (g_idx + 1) % 10 == 0:
            print(f"  Group {g_idx + 1}/{groups.num_groups()}, "
                  f"unique Paulis: {len(pauli_expectations)}")

    # Total variance and shot count
    sum_sigma_g = sum(per_group_sigma)
    total_variance = sum(per_group_variance)
    shots = int(np.ceil(sum_sigma_g ** 2 / (target_stderr ** 2)))

    return VarianceResult(
        total_variance=total_variance,
        sum_sigma_g=sum_sigma_g,
        shots_for_target=shots,
        target_stderr=target_stderr,
        per_group_variance=per_group_variance,
        per_group_sigma=per_group_sigma,
        energy=total_energy,
        num_terms=total_terms,
        num_groups=groups.num_groups()
    )


def compute_shot_count(
    variance_or_sum_sigma: float,
    target_stderr: float = 0.003,
    is_sum_sigma: bool = True
) -> int:
    """Compute shot count for target precision."""
    if is_sum_sigma:
        # N = (Σσ_g)² / ε²
        return int(np.ceil(variance_or_sum_sigma ** 2 / (target_stderr ** 2)))
    else:
        # N = Var(H) / ε² (uniform allocation approximation)
        return int(np.ceil(variance_or_sum_sigma / (target_stderr ** 2)))
