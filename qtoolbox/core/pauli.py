"""Symplectic representation of Pauli strings.

Pauli operators are encoded as two bit vectors (x_bits, z_bits):
- I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


class PauliString:
    """Symplectic representation of a Pauli string: coefficient * P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}

    Attributes:
        x_bits: Bitmap where bit i = 1 if P_i ∈ {X, Y}
        z_bits: Bitmap where bit i = 1 if P_i ∈ {Y, Z}
        coeff: Complex coefficient
        n_qubits: Number of qubits
    """

    __slots__ = ('x_bits', 'z_bits', 'coeff', 'n_qubits')

    def __init__(self, x_bits: int, z_bits: int, coeff: complex = 1.0, n_qubits: Optional[int] = None):
        """Initialize from symplectic representation. n_qubits inferred from bit length if None."""
        self.x_bits = x_bits
        self.z_bits = z_bits
        # Keep coefficient as-is (float or complex) instead of forcing to complex
        # This allows proper storage of real coefficients from hermitianized Hamiltonians
        self.coeff = coeff

        if n_qubits is None:
            # Infer from maximum bit position
            max_bit = max(x_bits, z_bits)
            self.n_qubits = max_bit.bit_length() if max_bit > 0 else 0
        else:
            self.n_qubits = n_qubits

    @classmethod
    def identity(cls, n_qubits: int, coeff: complex = 1.0) -> 'PauliString':
        """Create an identity operator on n qubits."""
        return cls(0, 0, coeff, n_qubits)

    @classmethod
    def from_string(cls, pauli_str: str, coeff: complex = 1.0) -> 'PauliString':
        """Create PauliString from string notation (e.g., "IXYZ")."""
        n_qubits = len(pauli_str)
        x_bits = 0
        z_bits = 0

        for i, p in enumerate(pauli_str):
            if p == 'X':
                x_bits |= (1 << i)
            elif p == 'Y':
                x_bits |= (1 << i)
                z_bits |= (1 << i)
            elif p == 'Z':
                z_bits |= (1 << i)
            elif p != 'I':
                raise ValueError(f"Invalid Pauli character: {p}. Must be I, X, Y, or Z.")

        return cls(x_bits, z_bits, coeff, n_qubits)

    @classmethod
    def from_dict(cls, pauli_dict: Dict[int, str], n_qubits: int, coeff: complex = 1.0) -> 'PauliString':
        """Create from dictionary {qubit_index: 'X'/'Y'/'Z'}. Identity qubits omitted."""
        x_bits = 0
        z_bits = 0

        for qubit, pauli in pauli_dict.items():
            if qubit >= n_qubits:
                raise ValueError(f"Qubit index {qubit} >= n_qubits {n_qubits}")

            if pauli == 'X':
                x_bits |= (1 << qubit)
            elif pauli == 'Y':
                x_bits |= (1 << qubit)
                z_bits |= (1 << qubit)
            elif pauli == 'Z':
                z_bits |= (1 << qubit)
            else:
                raise ValueError(f"Invalid Pauli: {pauli}. Must be X, Y, or Z.")

        return cls(x_bits, z_bits, coeff, n_qubits)

    def to_string(self) -> str:
        """Convert to string notation (e.g., 'IXYZ')."""
        result = []
        for i in range(self.n_qubits):
            x_bit = (self.x_bits >> i) & 1
            z_bit = (self.z_bits >> i) & 1

            if x_bit and z_bit:
                result.append('Y')
            elif x_bit:
                result.append('X')
            elif z_bit:
                result.append('Z')
            else:
                result.append('I')

        return ''.join(result)

    def to_dict(self) -> Dict[int, str]:
        """Convert to dictionary {qubit_index: 'X'/'Y'/'Z'}. Identity qubits omitted."""
        result = {}
        for i in range(self.n_qubits):
            x_bit = (self.x_bits >> i) & 1
            z_bit = (self.z_bits >> i) & 1

            if x_bit and z_bit:
                result[i] = 'Y'
            elif x_bit:
                result[i] = 'X'
            elif z_bit:
                result[i] = 'Z'

        return result

    def weight(self) -> int:
        """Return the weight (number of non-identity Paulis)."""
        # Count bits set in either x_bits or z_bits
        return bin(self.x_bits | self.z_bits).count('1')

    def support(self) -> int:
        """Return bitmap of qubits with non-identity operators."""
        return self.x_bits | self.z_bits

    def commutes_with(self, other: 'PauliString') -> bool:
        """Check if this Pauli commutes with another using symplectic inner product."""
        inner_product = (self.x_bits & other.z_bits) ^ (self.z_bits & other.x_bits)
        # Count number of 1 bits (mod 2)
        return bin(inner_product).count('1') % 2 == 0

    def k_commutes(self, other: 'PauliString', k: int) -> bool:
        """Check if operators k-commute."""
        n_blocks = (self.n_qubits + k - 1) // k

        for block_idx in range(n_blocks):
            start = block_idx * k
            end = min(start + k, self.n_qubits)

            # Create mask for qubits [start, end)
            block_mask = ((1 << end) - 1) ^ ((1 << start) - 1)

            # Compute inner product on this block
            inner_product = (self.x_bits & other.z_bits) ^ (self.z_bits & other.x_bits)
            inner_product &= block_mask

            if bin(inner_product).count('1') % 2 != 0:
                return False

        return True

    def multiply(self, other: 'PauliString') -> 'PauliString':
        """Multiply two Pauli strings, tracking phase correctly.

        Phase tracking follows the rules:
        - I*P = P, X*X = Y*Y = Z*Z = I
        - X*Y = iZ, Y*Z = iX, Z*X = iY (cyclic)
        - Y*X = -iZ, Z*Y = -iX, X*Z = -iY (anti-cyclic)
        """
        # XOR for Pauli multiplication in symplectic representation
        new_x = self.x_bits ^ other.x_bits
        new_z = self.z_bits ^ other.z_bits

        # Compute phase: coefficient product and commutator phase
        phase = self.coeff * other.coeff

        # Track phase from anti-commutation
        # Phase is i^power where power counts certain bit patterns
        power = 0
        for q in range(max(self.n_qubits, other.n_qubits)):
            x1 = (self.x_bits >> q) & 1
            z1 = (self.z_bits >> q) & 1
            x2 = (other.x_bits >> q) & 1
            z2 = (other.z_bits >> q) & 1

            # Compute phase contribution from this qubit
            # Uses lookup table for Pauli multiplication phases
            pauli1 = 2 * z1 + x1  # 0=I, 1=X, 2=Z, 3=Y
            pauli2 = 2 * z2 + x2

            # Phase table: phases[p1][p2] gives i^k for P1*P2 = i^k * P3
            # Encoding: 2*z + x maps to I=0, X=1, Z=2, Y=3
            # From matrix multiplication:
            # I*P = P,      X*I=X,  X*X=I,   X*Z=-iY, X*Y=iZ
            # Z*I=Z, Z*X=iY, Z*Z=I, Z*Y=-iX
            # Y*I=Y, Y*X=-iZ, Y*Z=iX, Y*Y=I
            phases = [
                [0, 0, 0, 0],   # I * {I,X,Z,Y}
                [0, 0, 3, 1],   # X * {I,X,Z,Y} -> {X, I, -iY, iZ}
                [0, 1, 0, 3],   # Z * {I,X,Z,Y} -> {Z, iY, I, -iX}
                [0, 3, 1, 0]    # Y * {I,X,Z,Y} -> {Y, -iZ, iX, I}
            ]
            power += phases[pauli1][pauli2]

        # Apply i^power to phase
        phase *= (1j) ** (power % 4)

        n_qubits = max(self.n_qubits, other.n_qubits)
        return PauliString(new_x, new_z, phase, n_qubits)

    def __eq__(self, other: object) -> bool:
        """Check equality of Pauli operators (ignoring coefficient)."""
        if not isinstance(other, PauliString):
            return False
        return (self.x_bits == other.x_bits and
                self.z_bits == other.z_bits and
                self.n_qubits == other.n_qubits)

    def __hash__(self) -> int:
        return hash((self.x_bits, self.z_bits, self.n_qubits))

    def __repr__(self) -> str:
        coeff_str = f"{self.coeff:.3f}" if abs(self.coeff - 1.0) > 1e-10 else ""
        pauli_str = self.to_string()
        if coeff_str:
            return f"{coeff_str} * {pauli_str}"
        return pauli_str

    def __str__(self) -> str:
        return self.__repr__()

    def copy(self) -> 'PauliString':
        return PauliString(self.x_bits, self.z_bits, self.coeff, self.n_qubits)


def symplectic_inner_product(x1: int, z1: int, x2: int, z2: int) -> int:
    """Compute symplectic inner product. Returns 0 if commute, 1 if anticommute."""
    inner_product = (x1 & z2) ^ (z1 & x2)
    return bin(inner_product).count('1') % 2
