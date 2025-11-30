"""Group representation for Pauli operators."""

from typing import List, Optional, Dict, Tuple
import numpy as np
import pickle
import cirq

from qtoolbox.core.pauli import PauliString


class PauliGroup:
    """A group of mutually commuting Pauli operators."""

    def __init__(self):
        self.paulis: List[PauliString] = []
        # Track symplectic representatives for fast commutation checks
        self.gx = 0
        self.gz = 0
        self.valid = 0

    def can_add(self, pauli: PauliString) -> bool:
        """Check if pauli commutes with all members."""
        return all(pauli.commutes_with(p) for p in self.paulis)

    def add(self, pauli: PauliString):
        """Add pauli to the group and update representatives."""
        self.paulis.append(pauli)
        # Update group representatives
        self.gx |= pauli.x_bits
        self.gz |= pauli.z_bits
        # Note: valid field is NOT updated here - caller must manage it

    def size(self) -> int:
        return len(self.paulis)

    def variance(self) -> float:
        """Compute variance for this group assuming uniform shots."""
        var = 0.0
        for p in self.paulis:
            var += abs(p.coeff) ** 2
        return var


class GroupCollection:
    """Collection of Pauli groups."""

    def __init__(self):
        self.groups: List[PauliGroup] = []

    def num_groups(self) -> int:
        return len(self.groups)

    def total_terms(self) -> int:
        """Total number of Pauli terms across all groups."""
        return sum(g.size() for g in self.groups)

    def shot_count_uniform(self, total_shots: int) -> float:
        """Compute worst-case variance bound with uniform shot allocation."""
        shots_per_group = total_shots / self.num_groups()

        # Variance = sum_i c_i^2 / N_i
        # where N_i is number of groups measuring P_i

        # Track how many times each Pauli is measured
        pauli_measurement_count = {}
        for group in self.groups:
            for p in group.paulis:
                pauli_measurement_count[p] = pauli_measurement_count.get(p, 0) + 1

        # Compute variance
        variance = 0.0
        seen = set()
        for p in pauli_measurement_count.keys():
            if p not in seen:
                n_measurements = pauli_measurement_count[p]
                variance += abs(p.coeff) ** 2 / (n_measurements * shots_per_group)
                seen.add(p)

        return variance

    def shot_count_optimal(self, total_shots: int) -> float:
        """Compute variance with optimal shot allocation (proportional to sqrt(variance))."""
        group_variances = [g.variance() for g in self.groups]
        sum_sqrt_var = sum(np.sqrt(v) for v in group_variances)

        total_variance = 0.0
        for var in group_variances:
            # Optimal shots: N_i = N * sqrt(var_i) / sum(sqrt(var_j))
            shots_i = total_shots * np.sqrt(var) / sum_sqrt_var
            total_variance += var / shots_i

        return total_variance

    def shot_count_optimal_cvx(
        self,
        total_shots: int,
        warm_start: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> tuple:
        """Compute optimal shot allocation for overlapping groups using convex optimization."""
        from scipy.optimize import minimize

        num_groups = self.num_groups()

        # Build pauli-to-groups mapping and extract coefficients
        # CRITICAL: Use tuple keys instead of PauliString objects!
        # PauliString.__eq__ and __hash__ ignore coefficients, causing collisions.
        pauli_to_groups = {}  # maps (x_bits, z_bits, coeff, n_qubits) -> list of group indices
        pauli_key_to_obj = {}  # maps tuple key -> first PauliString object we see
        all_paulis = []

        for g_idx, group in enumerate(self.groups):
            for p in group.paulis:
                key = (p.x_bits, p.z_bits, p.coeff, p.n_qubits)
                if key not in pauli_to_groups:
                    pauli_to_groups[key] = []
                    pauli_key_to_obj[key] = p
                    all_paulis.append(p)
                pauli_to_groups[key].append(g_idx)

        num_paulis = len(all_paulis)

        # Build sparse A matrix: A[i, g] = 1 if pauli i is in group g
        # For efficiency, store as dict of dicts: A[i][g] = 1
        A = {}
        coeffs_sq = np.zeros(num_paulis)

        for i, p in enumerate(all_paulis):
            coeffs_sq[i] = abs(p.coeff) ** 2
            A[i] = {}
            key = (p.x_bits, p.z_bits, p.coeff, p.n_qubits)
            for g in pauli_to_groups[key]:
                A[i][g] = 1.0

        # Objective function: f(n) = sum_i c_i^2 / (sum_g A[i,g] * n_g)
        def objective(n):
            total_var = 0.0
            for i in range(num_paulis):
                total_shots_i = sum(A[i].get(g, 0) * n[g] for g in A[i])
                if total_shots_i > 1e-12:  # Avoid division by zero
                    total_var += coeffs_sq[i] / total_shots_i
                else:
                    # Penalty for zero shots (should not happen with constraints)
                    total_var += 1e10 * coeffs_sq[i]
            return total_var

        # Gradient: df/dn_g = -sum_i c_i^2 * A[i,g] / (sum_j A[i,j] * n_j)^2
        def gradient(n):
            grad = np.zeros(num_groups)
            for i in range(num_paulis):
                total_shots_i = sum(A[i].get(g, 0) * n[g] for g in A[i])
                if total_shots_i > 1e-12:
                    factor = -coeffs_sq[i] / (total_shots_i ** 2)
                    for g in A[i]:
                        grad[g] += factor * A[i][g]
            return grad

        # Initial guess
        if warm_start is not None:
            if len(warm_start) != num_groups:
                raise ValueError(f"warm_start must have length {num_groups}, got {len(warm_start)}")
            # Normalize to match total_shots
            x0 = warm_start * (total_shots / warm_start.sum())
        else:
            # Uniform allocation
            x0 = np.full(num_groups, total_shots / num_groups)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda n: np.sum(n) - total_shots}  # sum(n_g) = total_shots
        ]

        # Bounds: n_g >= 0
        bounds = [(0, None) for _ in range(num_groups)]

        # Solve
        # Use tight tolerances for accurate convergence
        # ftol is absolute tolerance on objective function
        # For variance ~1e-3, we want relative precision ~1e-10, so ftol ~ 1e-13
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-15, 'eps': 1e-10, 'disp': verbose, 'maxiter': 10000}
        )

        if not result.success:
            import warnings
            warnings.warn(f"Optimization did not converge: {result.message}")

        optimal_shots = result.x
        optimal_variance = result.fun

        return optimal_shots, optimal_variance

    def compute_measured_variance(
        self,
        state: np.ndarray,
        qubits: List[cirq.Qid],
        shots_per_group: Optional[int] = None
    ) -> float:
        """Compute variance from quantum state including covariances."""
        from qtoolbox.converters.cirq_bridge import to_cirq
        from qtoolbox.measurement.diagonalization import diagonalize_pauli_group

        # Build list of all unique Paulis and which groups they appear in
        pauli_to_groups = {}
        all_paulis = []
        for g_idx, group in enumerate(self.groups):
            for p in group.paulis:
                if p not in pauli_to_groups:
                    pauli_to_groups[p] = []
                    all_paulis.append(p)
                pauli_to_groups[p].append(g_idx)

        # Compute expectation values for all Paulis
        n_qubits = len(qubits)
        expectations = {}
        for p in all_paulis:
            cirq_p = to_cirq(p, qubits)
            # Compute expectation value using cirq's PauliString expectation
            cirq_p_unit = cirq_p / cirq_p.coefficient  # Remove coefficient
            exp_val = cirq_p_unit.expectation_from_state_vector(state, {q: i for i, q in enumerate(qubits)})
            expectations[p] = np.real(exp_val)

        # Compute covariances between Paulis in the same group
        variance = 0.0
        for group in self.groups:
            # Covariance matrix for this group
            n_paulis = len(group.paulis)
            cov_matrix = np.zeros((n_paulis, n_paulis))

            for i, p_i in enumerate(group.paulis):
                for j, p_j in enumerate(group.paulis):
                    if i == j:
                        # Variance: Var(P_i) = <P_i^2> - <P_i>^2 = 1 - <P_i>^2
                        cov_matrix[i, i] = 1.0 - expectations[p_i] ** 2
                    else:
                        # Covariance: Cov(P_i, P_j) = <P_i P_j> - <P_i><P_j>
                        # For commuting Paulis, P_i P_j is also a Pauli
                        product = p_i.multiply(p_j)
                        if product in expectations:
                            exp_product = expectations[product]
                        else:
                            # Compute on the fly
                            cirq_product = to_cirq(product, qubits)
                            cirq_product_unit = cirq_product / cirq_product.coefficient
                            exp_product = np.real(cirq_product_unit.expectation_from_state_vector(state, {q: i for i, q in enumerate(qubits)}))
                            expectations[product] = exp_product

                        cov_matrix[i, j] = np.real(exp_product) - expectations[p_i] * expectations[p_j]

            # Compute variance contribution from this group
            # Var(sum c_i P_i) = sum_ij c_i c_j Cov(P_i, P_j)
            coeffs = np.array([p.coeff for p in group.paulis])
            group_var = coeffs @ cov_matrix @ coeffs

            # Account for finite shots if specified
            if shots_per_group is not None:
                group_var /= shots_per_group

            variance += group_var

        return variance

    def compute_measured_variance_repacking(
        self,
        state: np.ndarray,
        qubits: List[cirq.Qid],
        shots_per_group: Optional[int] = None
    ) -> float:
        """Compute variance for repacked groups accounting for multiple measurements."""
        from qtoolbox.converters.cirq_bridge import to_cirq

        # Track which groups measure each Pauli
        pauli_to_groups = {}
        all_paulis = []
        for g_idx, group in enumerate(self.groups):
            for p in group.paulis:
                if p not in pauli_to_groups:
                    pauli_to_groups[p] = []
                    all_paulis.append(p)
                pauli_to_groups[p].append(g_idx)

        # Compute variance for each unique Pauli
        variance = 0.0
        for p in all_paulis:
            # Skip Paulis with zero coefficient (don't contribute to variance)
            if abs(p.coeff) < 1e-14:
                continue

            cirq_p = to_cirq(p, qubits)
            # Normalize to unit coefficient for expectation value computation
            cirq_p_unit = cirq_p / cirq_p.coefficient
            exp_val = cirq_p_unit.expectation_from_state_vector(state, {q: i for i, q in enumerate(qubits)})
            exp_val = np.real(exp_val)

            # Variance of this Pauli measurement
            pauli_var = 1.0 - exp_val ** 2

            # Number of groups measuring this Pauli
            n_measurements = len(pauli_to_groups[p])

            # Variance contribution: c^2 * Var(P) / (N_measurements * shots_per_group)
            contrib = abs(p.coeff) ** 2 * pauli_var / n_measurements
            if shots_per_group is not None:
                contrib /= shots_per_group

            variance += contrib

        return variance

    def compute_variance_from_measurements(
        self,
        group_counts: List,
        shots_per_group: np.ndarray,
        qubits: List,
        include_covariances: bool = True
    ) -> Tuple[float, Dict]:
        """Compute variance from circuit measurement counts."""
        from qtoolbox.measurement.variance import estimate_variance_from_measurements
        return estimate_variance_from_measurements(
            self, group_counts, shots_per_group, qubits, include_covariances
        )

    @classmethod
    def load_symplectic(cls, filepath: str) -> 'GroupCollection':
        """Load GroupCollection from symplectic format pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Validate format
        required_keys = ['num_groups', 'x_bits', 'z_bits', 'coefficients']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Invalid symplectic format: missing key '{key}'")

        # Reconstruct GroupCollection
        collection = cls()

        # Infer n_qubits from maximum support across all groups
        n_qubits = 0
        for group_idx in range(data['num_groups']):
            x_bits = data['x_bits'][group_idx]
            z_bits = data['z_bits'][group_idx]

            for x, z in zip(x_bits, z_bits):
                support = x | z
                if support > 0:
                    n_qubits = max(n_qubits, int(support).bit_length())

        # If all terms are identity, use n_qubits=1 as default
        if n_qubits == 0:
            n_qubits = 1

        # Now reconstruct groups with correct n_qubits
        for group_idx in range(data['num_groups']):
            group = PauliGroup()

            x_bits = data['x_bits'][group_idx]
            z_bits = data['z_bits'][group_idx]
            coeffs = data['coefficients'][group_idx]

            # Reconstruct PauliStrings
            for x, z, coeff in zip(x_bits, z_bits, coeffs):
                # Convert to real if imaginary part is negligible (from hermitianization)
                # This matches the behavior of Hamiltonian.hermitianize()
                if abs(np.imag(coeff)) < 1e-15:
                    coeff_val = float(np.real(coeff))
                else:
                    coeff_val = complex(coeff)

                pauli = PauliString(
                    x_bits=int(x),
                    z_bits=int(z),
                    coeff=coeff_val,
                    n_qubits=n_qubits
                )
                group.add(pauli)

            collection.groups.append(group)

        return collection

    def __repr__(self) -> str:
        return f"GroupCollection({self.num_groups()} groups, {self.total_terms()} terms)"

    def __str__(self) -> str:
        return self.__repr__()
