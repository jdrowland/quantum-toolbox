"""Hamiltonian loaders for different file formats."""

import numpy as np
import logging
from typing import Optional
import openfermion as of

from qtoolbox.core.hamiltonian import Hamiltonian, HamiltonianLoader
from qtoolbox.core.pauli import PauliString
from qtoolbox.converters.openfermion_bridge import from_openfermion

logger = logging.getLogger(__name__)


class HDF5Loader(HamiltonianLoader):
    """Load Hamiltonian from HDF5 file (OpenFermion MolecularData format)."""

    def __init__(self, filepath: str, hermitianize: bool = True):
        self.filepath = filepath
        self.hermitianize = hermitianize

    def load(self) -> Hamiltonian:
        """Load Hamiltonian from HDF5 file."""
        mol_data = of.MolecularData(filename=self.filepath)
        mol_ham = mol_data.get_molecular_hamiltonian()

        fermion_ham = of.get_fermion_operator(mol_ham)
        qubit_ham = of.jordan_wigner(fermion_ham)

        terms = []
        for pauli_term, coeff in qubit_ham.terms.items():
            n_qubits = of.count_qubits(qubit_ham)
            terms.append(from_openfermion(pauli_term, coeff, n_qubits))

        metadata = {
            'source': 'HDF5',
            'filepath': self.filepath,
            'molecule': mol_data.name if hasattr(mol_data, 'name') else None,
            'n_electrons': mol_data.n_electrons if hasattr(mol_data, 'n_electrons') else None,
            'n_orbitals': mol_data.n_orbitals if hasattr(mol_data, 'n_orbitals') else None,
        }

        ham = Hamiltonian(terms, metadata)

        # Apply hermitianization if requested
        if self.hermitianize:
            n_before = ham.num_terms()
            ham, removed = ham.hermitianize()

            if removed:
                logger.warning(
                    f"Hermitianization removed {len(removed)} term(s) with purely "
                    f"imaginary coefficients from {self.filepath}:"
                )
                for pauli, coeff in removed[:5]:
                    logger.warning(f"  {pauli}: {coeff}")
                if len(removed) > 5:
                    logger.warning(f"  ... and {len(removed) - 5} more")

                # Sanity check: if we removed more than 1% of terms, something is wrong!
                removal_pct = 100 * len(removed) / n_before
                if removal_pct > 1.0:
                    logger.error(
                        f"WARNING: Removed {removal_pct:.1f}% of terms during hermitianization! "
                        f"This suggests a serious problem with the Hamiltonian."
                    )

        return ham


class NPZLoader(HamiltonianLoader):
    """Load Hamiltonian from NPZ file (custom format with molecular integrals)."""

    def __init__(self, filepath: str, hermitianize: bool = True):
        self.filepath = filepath
        self.hermitianize = hermitianize

    def load(self) -> Hamiltonian:
        data = np.load(self.filepath)

        # Try multiple key formats (different OWP files use different conventions)
        ecore = data.get("ECORE", data.get("ecore", data.get("e_nuc")))
        if ecore is None:
            raise KeyError(f"Could not find core energy in {self.filepath}. Available keys: {list(data.keys())}")
        ecore = float(ecore)

        h1 = data.get("H1", data.get("h1"))
        h2 = data.get("H2", data.get("h2"))

        # norb might be missing - infer from h1 shape
        norb = data.get("NORB", data.get("norb"))
        if norb is None:
            norb = h1.shape[0]
        norb = int(norb)

        nelec = data.get("NELEC", data.get("nelec", data.get("nelectron")))
        if nelec is None:
            raise KeyError(f"Could not find electron count in {self.filepath}. Available keys: {list(data.keys())}")
        nelec = int(nelec)

        # Convert to spin-orbital basis
        h2_reordered = 0.5 * np.asarray(h2.transpose(0, 2, 3, 1), order="C")
        h1_spinorb, h2_spinorb = of.chem.molecular_data.spinorb_from_spatial(h1, h2_reordered)

        # Create InteractionOperator and convert
        interaction_op = of.InteractionOperator(ecore, h1_spinorb, h2_spinorb)
        fermion_ham = of.get_fermion_operator(interaction_op)
        qubit_ham = of.jordan_wigner(fermion_ham)

        terms = []
        n_qubits = of.count_qubits(qubit_ham)
        for pauli_term, coeff in qubit_ham.terms.items():
            terms.append(from_openfermion(pauli_term, coeff, n_qubits))

        metadata = {
            'source': 'NPZ',
            'filepath': self.filepath,
            'n_orbitals': norb,
            'n_electrons': nelec,
            'ecore': ecore,
        }

        ham = Hamiltonian(terms, metadata)

        # Apply hermitianization if requested
        if self.hermitianize:
            n_before = ham.num_terms()
            ham, removed = ham.hermitianize()

            if removed:
                logger.warning(
                    f"Hermitianization removed {len(removed)} term(s) with purely "
                    f"imaginary coefficients from {self.filepath}:"
                )
                for pauli, coeff in removed[:5]:
                    logger.warning(f"  {pauli}: {coeff}")
                if len(removed) > 5:
                    logger.warning(f"  ... and {len(removed) - 5} more")

                # Sanity check: if we removed more than 1% of terms, something is wrong!
                removal_pct = 100 * len(removed) / n_before
                if removal_pct > 1.0:
                    logger.error(
                        f"WARNING: Removed {removal_pct:.1f}% of terms during hermitianization! "
                        f"This suggests a serious problem with the Hamiltonian."
                    )

        return ham


class FermiHubbardLoader(HamiltonianLoader):
    """Generate Fermi-Hubbard Hamiltonian programmatically."""

    def __init__(
        self,
        x_dimension: int,
        y_dimension: int,
        tunneling: float = 1.0,
        coulomb: float = 4.0,
        chemical_potential: float = 0.0,
        magnetic_field: float = 0.0,
        periodic: bool = True,
        spinless: bool = False,
    ):
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.tunneling = tunneling
        self.coulomb = coulomb
        self.chemical_potential = chemical_potential
        self.magnetic_field = magnetic_field
        self.periodic = periodic
        self.spinless = spinless

    def load(self) -> Hamiltonian:
        fermion_ham = of.fermi_hubbard(
            x_dimension=self.x_dimension,
            y_dimension=self.y_dimension,
            tunneling=self.tunneling,
            coulomb=self.coulomb,
            chemical_potential=self.chemical_potential,
            magnetic_field=self.magnetic_field,
            periodic=self.periodic,
            spinless=self.spinless,
        )

        qubit_ham = of.jordan_wigner(fermion_ham)

        terms = []
        n_qubits = of.count_qubits(qubit_ham)
        for pauli_term, coeff in qubit_ham.terms.items():
            terms.append(from_openfermion(pauli_term, coeff, n_qubits))

        metadata = {
            'source': 'FermiHubbard',
            'x_dimension': self.x_dimension,
            'y_dimension': self.y_dimension,
            'tunneling': self.tunneling,
            'coulomb': self.coulomb,
            'periodic': self.periodic,
            'spinless': self.spinless,
        }

        return Hamiltonian(terms, metadata)
