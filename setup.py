from setuptools import setup, find_packages

setup(
    name="quantum-toolbox",
    version="0.1.0",
    description="Unified library for quantum Hamiltonian grouping and measurement",
    author="Jeremiah Rowland",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "cirq>=1.0.0",
        "openfermion>=1.5.0",
        "h5py>=3.0.0",
        "quimb>=1.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "qsim": ["qsimcirq>=0.14.0"],
        "qiskit": ["qiskit>=0.40.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
    },
)
