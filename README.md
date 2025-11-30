# Quantum Toolbox

A unified library for quantum Hamiltonian grouping, measurement, and variance estimation.

## Features

- **Symplectic Pauli representation**: Fast bitwise commutation checks
- **Hamiltonian loading**: HDF5, NPZ, and Fermi-Hubbard generation
- **Grouping algorithms**: Sorted insertion, ad-hoc repacking, post-hoc repacking
- **Variance estimation**: Shot-weighted averaging with covariance handling
- **DMRG integration**: MPS state preparation via quimb
- **Multiple backends**: Cirq, qsimcirq (GPU), Qiskit, Quimb

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from qtoolbox.core import HDF5Loader, GroupCollection
from qtoolbox.grouping import sorted_insertion_grouping

# Load Hamiltonian
hamiltonian = HDF5Loader("molecule.hdf5").load()

# Group terms
groups = sorted_insertion_grouping(hamiltonian)

print(f"Grouped {hamiltonian.num_terms()} terms into {groups.num_groups()} groups")
```

## Project Structure

- `qtoolbox/core/`: Core data structures (PauliString, Hamiltonian, GroupCollection)
- `qtoolbox/grouping/`: Grouping algorithms (sorted insertion, repacking)
- `qtoolbox/measurement/`: Variance estimation and simulation
- `qtoolbox/converters/`: Cirq and OpenFermion bridges
- `qtoolbox/states/`: DMRG and MPS utilities
