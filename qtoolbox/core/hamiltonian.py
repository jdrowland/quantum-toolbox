"""Hamiltonian representation and loaders."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qtoolbox.core.pauli import PauliString


class Hamiltonian:
    """Hamiltonian as a sum of Pauli strings."""

    def __init__(self, terms: List[PauliString], metadata: Optional[Dict[str, Any]] = None):
        if not terms:
            raise ValueError("Hamiltonian must have at least one term")

        self.terms = terms
        self.metadata = metadata or {}

        # Verify all terms have same n_qubits
        n_qubits = terms[0].n_qubits
        if not all(t.n_qubits == n_qubits for t in terms):
            raise ValueError("All terms must have same n_qubits")

    def num_terms(self) -> int:
        return len(self.terms)

    def num_qubits(self) -> int:
        return self.terms[0].n_qubits

    def sort_by_coefficient(self, descending: bool = True) -> 'Hamiltonian':
        """Return new Hamiltonian with terms sorted by |coefficient|."""
        sorted_terms = sorted(self.terms, key=lambda t: abs(t.coeff), reverse=descending)
        return Hamiltonian(sorted_terms, self.metadata.copy())

    def sort_by_weight(self, ascending: bool = True) -> 'Hamiltonian':
        """Return new Hamiltonian with terms sorted by weight."""
        sorted_terms = sorted(self.terms, key=lambda t: t.weight(), reverse=not ascending)
        return Hamiltonian(sorted_terms, self.metadata.copy())

    def get_term(self, idx: int) -> PauliString:
        """Get term by index."""
        return self.terms[idx]

    def is_hermitian(self, tol: float = 1e-14) -> bool:
        """Check if Hamiltonian is Hermitian (all coefficients real)."""
        return all(abs(np.imag(term.coeff)) < tol for term in self.terms)

    def verify_hermitian(self, tol: float = 1e-14) -> None:
        """Verify Hamiltonian is Hermitian, raise ValueError if not."""
        non_hermitian = [(term, term.coeff) for term in self.terms
                        if abs(np.imag(term.coeff)) >= tol]

        if non_hermitian:
            error_msg = f"Hamiltonian has {len(non_hermitian)} terms with imaginary coefficients:\n"
            for term, coeff in non_hermitian[:5]:
                error_msg += f"  {term}: {coeff}\n"
            if len(non_hermitian) > 5:
                error_msg += f"  ... and {len(non_hermitian) - 5} more\n"
            raise ValueError(error_msg)

    def hermitianize(self, prune_tol: float = 1e-15) -> Tuple['Hamiltonian', List[Tuple[PauliString, complex]]]:
        """Return Hermitianized Hamiltonian: (H + H†) / 2."""
        hermitian_terms = []
        removed_terms = []

        for pauli in self.terms:
            # H† has conjugated coefficients (Pauli operators are self-adjoint)
            new_coeff = (pauli.coeff + np.conj(pauli.coeff)) / 2

            # For real coefficients: (c + c*)/2 = c
            # For purely imaginary: (ia + (-ia))/2 = 0

            if abs(new_coeff) > prune_tol:
                # Keep term with hermitianized coefficient
                # Convert to real if close enough (remove floating point noise)
                if abs(np.imag(new_coeff)) < prune_tol:
                    new_coeff = np.real(new_coeff)

                hermitian_terms.append(
                    PauliString(pauli.x_bits, pauli.z_bits, new_coeff, pauli.n_qubits)
                )
            else:
                # Track removed terms for logging
                removed_terms.append((pauli, pauli.coeff))

        if not hermitian_terms:
            raise ValueError("Hermitianization removed all terms! This suggests a serious problem.")

        return Hamiltonian(hermitian_terms, self.metadata.copy()), removed_terms

    def __repr__(self) -> str:
        return f"Hamiltonian({self.num_terms()} terms, {self.num_qubits()} qubits)"

    def __str__(self) -> str:
        return self.__repr__()


class HamiltonianLoader(ABC):
    """Abstract base for Hamiltonian loaders."""

    @abstractmethod
    def load(self) -> Hamiltonian:
        pass
