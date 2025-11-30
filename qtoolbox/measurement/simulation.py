"""Simulator backends for measurement simulation."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import cirq
import numpy as np

from qtoolbox.measurement.data import MeasurementData, EndianConvention


class SimulatorBackend(ABC):
    """Abstract base for simulator backends."""

    @abstractmethod
    def run(
        self,
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        shots: int
    ) -> MeasurementData:
        """Run circuit and return measurement data."""
        pass


class CirqSimulator(SimulatorBackend):
    """Cirq's default noiseless simulator."""

    def __init__(self):
        self.simulator = cirq.Simulator()

    def run(
        self,
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        shots: int
    ) -> MeasurementData:
        """Run circuit with Cirq simulator."""
        # Add measurements
        full_circuit = circuit.copy()
        full_circuit.append(cirq.measure(*qubits, key='result'))

        # Run
        result = self.simulator.run(full_circuit, repetitions=shots)
        measurements = result.measurements['result']

        # Convert to bitstring counts
        counts = {}
        for measurement in measurements:
            bitstring = ''.join(str(bit) for bit in measurement)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return MeasurementData(counts, EndianConvention.LITTLE)


class QsimSimulator(SimulatorBackend):
    """qsimcirq GPU-accelerated simulator."""

    def __init__(self, use_gpu: bool = True):
        try:
            import qsimcirq
            qsim_options = qsimcirq.QSimOptions(use_gpu=use_gpu, gpu_mode=0)
            self.simulator = qsimcirq.QSimSimulator(qsim_options=qsim_options)
            self.available = True
        except (ImportError, ValueError):
            self.simulator = cirq.Simulator()
            self.available = False

    def run(
        self,
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        shots: int
    ) -> MeasurementData:
        """Run circuit with qsimcirq."""
        full_circuit = circuit.copy()
        full_circuit.append(cirq.measure(*qubits, key='result'))

        result = self.simulator.run(full_circuit, repetitions=shots)
        measurements = result.measurements['result']

        counts = {}
        for measurement in measurements:
            bitstring = ''.join(str(bit) for bit in measurement)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return MeasurementData(counts, EndianConvention.LITTLE)


class NoisySimulator(SimulatorBackend):
    """Cirq simulator with depolarizing noise."""

    def __init__(self, depolarizing_p: float = 0.001):
        self.simulator = cirq.DensityMatrixSimulator()
        self.depolarizing_p = depolarizing_p

    def run(
        self,
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        shots: int
    ) -> MeasurementData:
        """Run circuit with noise."""
        # Add noise after each moment
        noisy_circuit = cirq.Circuit()
        for moment in circuit:
            noisy_circuit.append(moment)
            qubits_in_moment = set()
            for op in moment:
                qubits_in_moment.update(op.qubits)
            for q in qubits_in_moment:
                noisy_circuit.append(cirq.depolarize(self.depolarizing_p)(q))

        noisy_circuit.append(cirq.measure(*qubits, key='result'))

        result = self.simulator.run(noisy_circuit, repetitions=shots)
        measurements = result.measurements['result']

        counts = {}
        for measurement in measurements:
            bitstring = ''.join(str(bit) for bit in measurement)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return MeasurementData(counts, EndianConvention.LITTLE)


class QiskitSimulator(SimulatorBackend):
    """Qiskit Aer simulator with GPU support."""

    def __init__(self, use_gpu: bool = False, backend_name: str = "aer_simulator"):
        """Initialize Qiskit simulator.

        Args:
            use_gpu: Use GPU if available (requires qiskit-aer-gpu)
            backend_name: Aer backend to use
        """
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit_aer import AerSimulator

            self.available = True
            self.use_gpu = use_gpu

            if use_gpu:
                try:
                    self.backend = AerSimulator(method='statevector', device='GPU')
                except Exception:
                    self.backend = AerSimulator(method='statevector', device='CPU')
                    self.use_gpu = False
            else:
                self.backend = AerSimulator(method='statevector', device='CPU')

        except ImportError:
            self.available = False
            self.backend = None

    def run(
        self,
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        shots: int
    ) -> MeasurementData:
        """Run circuit with Qiskit Aer."""
        if not self.available:
            raise ImportError("Qiskit not available, falling back disabled")

        from qiskit import QuantumCircuit, transpile

        n_qubits = len(qubits)
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Convert Cirq circuit to Qiskit
        qubit_map = {q: i for i, q in enumerate(qubits)}

        for moment in circuit:
            for op in moment:
                if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1.0:
                    qc.x(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.YPowGate) and op.gate.exponent == 1.0:
                    qc.y(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.ZPowGate) and op.gate.exponent == 1.0:
                    qc.z(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.HPowGate) and op.gate.exponent == 1.0:
                    qc.h(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.CXPowGate) and op.gate.exponent == 1.0:
                    qc.cx(qubit_map[op.qubits[0]], qubit_map[op.qubits[1]])
                elif isinstance(op.gate, cirq.CZPowGate) and op.gate.exponent == 1.0:
                    qc.cz(qubit_map[op.qubits[0]], qubit_map[op.qubits[1]])
                else:
                    raise NotImplementedError(f"Gate {op.gate} not supported in Qiskit bridge")

        qc.measure(range(n_qubits), range(n_qubits))

        qc_transpiled = transpile(qc, self.backend)
        result = self.backend.run(qc_transpiled, shots=shots).result()
        counts = result.get_counts()

        # Qiskit uses big-endian
        return MeasurementData(counts, EndianConvention.BIG)


class QuimbTNSimulator(SimulatorBackend):
    """Quimb tensor network simulator."""

    def __init__(self, method: str = "mps", max_bond: Optional[int] = None,
                 backend: str = "numpy", device: str = "cpu"):
        """Initialize Quimb TN simulator.

        Args:
            method: TN method ('mps', 'dense', 'auto')
            max_bond: Maximum bond dimension for MPS
            backend: Array backend ('numpy', 'cupy', 'jax')
            device: Device for computation ('cpu', 'cuda')
        """
        try:
            import quimb.tensor as qtn
            self.available = True
            self.method = method
            self.max_bond = max_bond
            self.backend = backend
            self.device = device

            if backend == 'cupy' and device == 'cuda':
                try:
                    import cupy
                    self.use_gpu = True
                except ImportError:
                    self.backend = 'numpy'
                    self.device = 'cpu'
                    self.use_gpu = False
            else:
                self.use_gpu = False

        except ImportError:
            self.available = False

    def run(
        self,
        circuit: cirq.Circuit,
        qubits: List[cirq.Qid],
        shots: int
    ) -> MeasurementData:
        """Run circuit with Quimb TN."""
        if not self.available:
            raise ImportError("Quimb not available")

        import quimb.tensor as qtn

        n_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}

        # Build circuit in Quimb
        circ = qtn.Circuit(n_qubits)

        for moment in circuit:
            for op in moment:
                if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1.0:
                    circ.x(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.YPowGate) and op.gate.exponent == 1.0:
                    circ.y(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.ZPowGate) and op.gate.exponent == 1.0:
                    circ.z(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.HPowGate) and op.gate.exponent == 1.0:
                    circ.h(qubit_map[op.qubits[0]])
                elif isinstance(op.gate, cirq.CXPowGate) and op.gate.exponent == 1.0:
                    circ.cx(qubit_map[op.qubits[0]], qubit_map[op.qubits[1]])
                elif isinstance(op.gate, cirq.CZPowGate) and op.gate.exponent == 1.0:
                    circ.cz(qubit_map[op.qubits[0]], qubit_map[op.qubits[1]])
                else:
                    raise NotImplementedError(f"Gate {op.gate} not supported in Quimb bridge")

        # Get final state
        if self.method == 'mps':
            psi = circ.psi
            if self.max_bond:
                psi.compress(max_bond=self.max_bond)
        else:
            psi = circ.psi

        # Sample from state
        state_vector = psi.to_dense()
        if self.backend == 'cupy':
            import cupy as cp
            state_vector = cp.asnumpy(state_vector)

        # Flatten to 1D if needed
        state_vector = state_vector.flatten()

        probabilities = np.abs(state_vector) ** 2
        probabilities /= probabilities.sum()

        # Sample bitstrings
        samples = np.random.choice(2**n_qubits, size=shots, p=probabilities)
        counts = {}
        for sample in samples:
            bitstring = format(sample, f'0{n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Quimb uses big-endian by default
        return MeasurementData(counts, EndianConvention.BIG)


def get_simulator(
    backend: str = "cirq",
    use_gpu: bool = False,
    **kwargs
) -> SimulatorBackend:
    """Factory function to get appropriate simulator backend.

    Args:
        backend: Simulator backend ('cirq', 'qsim', 'qiskit', 'quimb', 'noisy')
        use_gpu: Use GPU acceleration if available
        **kwargs: Additional backend-specific options

    Returns:
        Configured simulator backend
    """
    if backend == "cirq":
        return CirqSimulator()
    elif backend == "qsim":
        return QsimSimulator(use_gpu=use_gpu)
    elif backend == "qiskit":
        return QiskitSimulator(use_gpu=use_gpu, **kwargs)
    elif backend == "quimb":
        device = "cuda" if use_gpu else "cpu"
        backend_str = kwargs.get("array_backend", "cupy" if use_gpu else "numpy")
        return QuimbTNSimulator(
            method=kwargs.get("method", "mps"),
            max_bond=kwargs.get("max_bond", None),
            backend=backend_str,
            device=device
        )
    elif backend == "noisy":
        return NoisySimulator(depolarizing_p=kwargs.get("depolarizing_p", 0.001))
    else:
        raise ValueError(f"Unknown backend: {backend}")
