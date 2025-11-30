"""Measurement data handling with endianness conventions."""

from enum import Enum
from typing import Dict


class EndianConvention(Enum):
    """Endianness conventions for bitstring ordering."""
    LITTLE = "little"  # Cirq: qubit 0 is rightmost bit
    BIG = "big"        # IBM/Qiskit: qubit 0 is leftmost bit


class MeasurementData:
    """Measurement counts with automatic endianness handling."""

    def __init__(self, counts: Dict[str, int], convention: EndianConvention):
        """Create MeasurementData with specified endianness convention.

        Internally stores in little-endian (Cirq) convention.
        """
        self._counts_little: Dict[str, int] = {}

        if convention == EndianConvention.LITTLE:
            self._counts_little = counts.copy()
        else:  # BIG
            # Reverse bitstrings to convert to little-endian
            self._counts_little = {bitstring[::-1]: count for bitstring, count in counts.items()}

    def get_counts(self, convention: EndianConvention) -> Dict[str, int]:
        """Get counts in specified endianness convention."""
        if convention == EndianConvention.LITTLE:
            return self._counts_little.copy()
        else:  # BIG
            return {bitstring[::-1]: count for bitstring, count in self._counts_little.items()}

    def total_shots(self) -> int:
        """Total number of shots."""
        return sum(self._counts_little.values())

    def __repr__(self) -> str:
        return f"MeasurementData({self.total_shots()} shots)"
