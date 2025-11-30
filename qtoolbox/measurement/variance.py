"""Variance estimation from circuit sampling measurements."""

import cirq
import numpy as np
from typing import Dict, List, Tuple
from qtoolbox.core.pauli import PauliString
from qtoolbox.measurement.data import MeasurementData, EndianConvention


class MeasurementSetup:
    """Stores measurement setup for a group: original and diagonalized Paulis."""
    def __init__(self, paulis: List[PauliString], diagonalized_paulis: List[PauliString], circuit: cirq.Circuit):
        self.paulis = paulis
        self.diagonalized_paulis = diagonalized_paulis
        self.circuit = circuit
        # Build mapping from (x_bits, z_bits, coeff) to diagonalized Pauli
        # Since __eq__ ignores coefficient, we need a tuple key
        self.pauli_map = {}
        for orig, diag in zip(paulis, diagonalized_paulis):
            key = (orig.x_bits, orig.z_bits, orig.coeff, orig.n_qubits)
            self.pauli_map[key] = diag


def compute_pauli_expectation_from_counts(
    pauli: PauliString,
    counts: Dict[str, int],
    qubits: List[cirq.Qid]
) -> float:
    """Compute expectation value from measurement counts for a diagonalized Pauli."""
    # Identity operator
    if pauli.x_bits == 0 and pauli.z_bits == 0:
        # Return coefficient for identity (not just 1.0)
        return float(np.real(pauli.coeff))

    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    # Get coefficient from diagonalized Pauli (includes Hamiltonian coeff AND Clifford phase)
    coefficient = float(np.real(pauli.coeff))

    expectation = 0.0
    for bitstring, count in counts.items():
        # Compute parity of Z-support
        parity = 0
        for i in range(pauli.n_qubits):
            if pauli.z_bits & (1 << i):
                bit = int(bitstring[i])
                parity += bit
        eigenvalue = (-1) ** (parity % 2)
        expectation += eigenvalue * count

    return coefficient * expectation / total_shots


def estimate_variance_from_measurements(
    measurement_setups: List[MeasurementSetup],
    group_counts: List[Dict[str, int]],
    shots_per_group: np.ndarray,
    qubits: List[cirq.Qid],
    include_covariances: bool = True
) -> Tuple[float, Dict]:
    """Estimate energy variance from measurement counts using shot-weighted averaging."""
    # Build pauli-to-groups mapping using original Paulis
    # IMPORTANT: We must include ALL Paulis from ALL groups for correct energy calculation,
    # even if some groups get 0 shots. We'll just track which groups have measurements.
    #
    # CRITICAL FIX: PauliString.__eq__ and __hash__ ignore the coefficient!
    # This causes Paulis with same (x,z) but different coefficients to collide.
    # We must use tuple keys that include the coefficient.
    pauli_to_groups = {}
    pauli_key_to_obj = {}  # Map tuple key to first Pauli object we see
    all_paulis = []

    for g_idx, setup in enumerate(measurement_setups):
        for p in setup.paulis:
            # Use tuple key that includes coefficient
            key = (p.x_bits, p.z_bits, p.coeff, p.n_qubits)
            if key not in pauli_to_groups:
                pauli_to_groups[key] = []
                pauli_key_to_obj[key] = p
                all_paulis.append(p)
            pauli_to_groups[key].append(g_idx)

    # ========================================================================
    # PASS 1: Shot-weighted expectation values
    # ========================================================================
    pauli_expectations = {}  # tuple_key -> final weighted expectation
    pauli_group_expectations = {}  # tuple_key -> {group_idx: expectation}
    pauli_total_shots = {}  # tuple_key -> total shots across all groups

    for p_key in pauli_to_groups.keys():
        p = pauli_key_to_obj[p_key]  # Get the Pauli object
        group_indices = pauli_to_groups[p_key]

        total_shots = 0
        weighted_expectation = 0.0
        group_exps = {}

        for g_idx in group_indices:
            shots_g = shots_per_group[g_idx]

            # Skip groups with 0 shots - only use measurements from groups that were actually measured
            if shots_g == 0:
                continue

            counts = group_counts[g_idx]
            setup = measurement_setups[g_idx]

            # Find the diagonalized Pauli corresponding to original Pauli p
            # Use the same tuple key
            diag_pauli = setup.pauli_map.get(p_key)

            if diag_pauli is None:
                raise ValueError(f"Pauli {p_key} not found in group {g_idx}")

            # Compute expectation using the diagonalized (Z-basis) Pauli
            exp_g = compute_pauli_expectation_from_counts(diag_pauli, counts, qubits)

            # Accumulate shot-weighted sum
            weighted_expectation += exp_g * shots_g
            total_shots += shots_g
            group_exps[g_idx] = exp_g

        # Final expectation: weighted average
        pauli_expectations[p_key] = weighted_expectation / total_shots if total_shots > 0 else 0.0
        pauli_total_shots[p_key] = total_shots
        pauli_group_expectations[p_key] = group_exps

    # ========================================================================
    # PASS 2: Variance computation with shot weighting
    # ========================================================================
    # For a Pauli measured in multiple groups, the variance is:
    # Var(<P>) = Σ_g w_g² Var(<P>_g)
    # where w_g = N_g / N_total is the shot weight
    # and Var(<P>_g) = (1 - <P>_g²) / N_g is the single-group variance
    #
    # EDGE CASE: If a Pauli is measured 0 times total (across all groups it appears in),
    # we assign infinite variance. This is critical for repacking where a Pauli can
    # appear in multiple groups - a Pauli should only get infinite variance if ALL
    # groups containing it get 0 shots, not just some groups.
    pauli_variances = {}

    for p_key in pauli_to_groups.keys():
        p = pauli_key_to_obj[p_key]
        total_shots = pauli_total_shots[p_key]
        if total_shots == 0:
            # Pauli measured 0 times total -> maximal variance
            # For a ±1 observable with no information, Var(P) = 1 (maximal)
            # Variance contribution to energy: c_i² * Var(P) = c_i² * 1
            coeff_magnitude = abs(float(np.real(p.coeff)))
            pauli_variances[p_key] = coeff_magnitude ** 2
            continue

        var = 0.0
        for g_idx, exp_g in pauli_group_expectations[p_key].items():
            shots_g = shots_per_group[g_idx]

            # Skip groups with 0 shots (shouldn't happen since we only add to
            # pauli_group_expectations for groups with measurements, but be safe)
            if shots_g == 0:
                continue

            weight_g = shots_g / total_shots

            # exp_g = c_i * <P_raw> where c_i is the coefficient
            # For variance of a ±1 observable, need raw expectation
            # Extract raw expectation by dividing by coefficient magnitude
            coeff_magnitude = abs(float(np.real(p.coeff)))
            if coeff_magnitude > 1e-15:
                exp_raw = exp_g / coeff_magnitude
            else:
                exp_raw = 0.0

            # Pauli variance: (1 - <P_raw>²) for a ±1 observable
            # Multiply by c_i² to get variance of c_i * P
            # Divided by shots for sampling variance
            var_g = coeff_magnitude ** 2 * (1.0 - exp_raw ** 2) / shots_g

            # Weighted contribution
            var += weight_g ** 2 * var_g

        pauli_variances[p_key] = var

    # ========================================================================
    # PASS 3: Covariance computation
    # ========================================================================
    # For Paulis measured together in the same group, there are correlations.
    # The covariance is:
    # Cov(<P_i>, <P_j>) = Σ_{g ∈ G_i ∩ G_j} w_{i,g} w_{j,g} Cov_g(P_i, P_j)
    # where Cov_g(P_i, P_j) = (E[P_i P_j] - E[P_i] E[P_j]) / N_g
    covariances = {}

    if include_covariances:
        for g_idx, setup in enumerate(measurement_setups):
            paulis_in_group = setup.paulis
            diag_paulis = setup.diagonalized_paulis

            # Skip groups with 0 or 1 Pauli (no covariances possible)
            if len(paulis_in_group) < 2:
                continue

            counts = group_counts[g_idx]
            shots_g = shots_per_group[g_idx]
            total_count = sum(counts.values())

            if total_count == 0 or shots_g == 0:
                continue

            # VECTORIZED APPROACH: Convert counts to numpy arrays once
            # Extract bitstrings and counts as arrays
            bitstrings = list(counts.keys())
            count_array = np.array([counts[bs] for bs in bitstrings])
            n_samples = len(bitstrings)

            # Convert bitstrings to integer array for vectorized parity computation
            # Shape: (n_samples, n_qubits)
            n_qubits = diag_paulis[0].n_qubits
            bitstring_matrix = np.array([[int(bs[q]) for q in range(n_qubits)]
                                         for bs in bitstrings], dtype=np.int8)

            # Precompute eigenvalues for all Paulis in this group
            # eigenvalues[i] = array of eigenvalues for pauli_i over all samples
            eigenvalues = []
            coefficients = []

            for diag_pauli in diag_paulis:
                # Create mask for Z-support (which qubits have Z operators)
                z_mask = np.array([(diag_pauli.z_bits >> q) & 1 for q in range(n_qubits)], dtype=np.int8)

                # Compute parity: sum of bits at Z positions, then (-1)^parity
                # Shape: (n_samples,)
                parity = (bitstring_matrix @ z_mask) % 2
                eigenval = (-1) ** parity

                eigenvalues.append(eigenval)
                coefficients.append(float(np.real(diag_pauli.coeff)))

            eigenvalues = np.array(eigenvalues)  # Shape: (n_paulis, n_samples)
            coefficients = np.array(coefficients)  # Shape: (n_paulis,)

            # Compute all pairwise covariances within this group
            for i, pauli_i in enumerate(paulis_in_group):
                # Convert to tuple key
                key_i = (pauli_i.x_bits, pauli_i.z_bits, pauli_i.coeff, pauli_i.n_qubits)
                if pauli_total_shots[key_i] == 0:
                    continue

                for j in range(i + 1, len(paulis_in_group)):
                    pauli_j = paulis_in_group[j]
                    key_j = (pauli_j.x_bits, pauli_j.z_bits, pauli_j.coeff, pauli_j.n_qubits)
                    if pauli_total_shots[key_j] == 0:
                        continue

                    # Shot weights for each Pauli
                    weight_i_g = shots_g / pauli_total_shots[key_i]
                    weight_j_g = shots_g / pauli_total_shots[key_j]

                    # Individual expectations for this group
                    exp_i_g = pauli_group_expectations[key_i][g_idx]
                    exp_j_g = pauli_group_expectations[key_j][g_idx]

                    # Vectorized computation of E[P_i * P_j]
                    # eigenvalues[i] and eigenvalues[j] are arrays of eigenvalues
                    # exp_product = sum(coeff_i * eigenval_i * coeff_j * eigenval_j * count) / total
                    product_eigenvalues = eigenvalues[i] * eigenvalues[j]  # Element-wise
                    exp_product = coefficients[i] * coefficients[j] * np.sum(product_eigenvalues * count_array) / total_count

                    # Covariance for this group: Cov_g = (E[XY] - E[X]E[Y]) / N_g
                    cov_g = (exp_product - exp_i_g * exp_j_g) / shots_g

                    # Weighted contribution to total covariance
                    weighted_cov = weight_i_g * weight_j_g * cov_g

                    # Accumulate (a pair may appear in multiple groups)
                    # Use tuple keys for covariance dict too
                    cov_key = (key_i, key_j)
                    if cov_key in covariances:
                        covariances[cov_key] += weighted_cov
                    else:
                        covariances[cov_key] = weighted_cov

    # ========================================================================
    # FINAL: Energy variance
    # ========================================================================
    # Var(H) = Σ_i c_i² Var(P_i) + 2 Σ_{i<j} c_i c_j Cov(P_i, P_j)
    # Note: pauli_variances[p_key] already contains c_i² * Var(<P>_raw), so just sum
    variance_contribution = 0.0
    for p_key in pauli_to_groups.keys():
        variance_contribution += pauli_variances[p_key]

    covariance_contribution = 0.0
    for (key_i, key_j), cov in covariances.items():
        # cov already contains Cov(c_i * P_i, c_j * P_j), so just sum with factor of 2
        covariance_contribution += 2.0 * cov

    total_variance = variance_contribution + covariance_contribution

    # Compute energy expectation: E = Σ_i c_i <P_i>
    # pauli_expectations[p_key] already contains c_i * <P_i> (diag coefficient includes both Hamiltonian coeff and Clifford phase)
    energy = sum(pauli_expectations[p_key] for p_key in pauli_to_groups.keys())

    # Package diagnostics
    diagnostics = {
        'energy': energy,
        'expectations': pauli_expectations,
        'variances': pauli_variances,
        'covariances': covariances,
        'variance_contribution': variance_contribution,
        'covariance_contribution': covariance_contribution,
    }

    return total_variance, diagnostics


def convert_measurement_data_to_counts(
    measurement_data: MeasurementData,
    convention: EndianConvention = EndianConvention.LITTLE
) -> Dict[str, int]:
    """Convert MeasurementData to a counts dictionary."""
    return measurement_data.get_counts(convention)
