"""For a given observable, compute its expectation value exactly and with the MPS code.
The MPO will be derived by converting a PauliString object into an MPO."""

import unittest
import numpy as np
from scipy.linalg import norm
import cirq
from quimb.tensor.tensor_1d import MatrixProductState
from qtoolbox.converters.quimb_bridge import to_quimb
from qtoolbox.converters.cirq_bridge import to_cirq, from_cirq
from qtoolbox.measurement.mps_variance import compute_pauli_expectation

class TestExpValue(unittest.TestCase):

    def test_two_qubits(self):
        nq = 2
        qs = cirq.LineQubit.range(nq)
        qubit_map = {q: i for i, q in enumerate(qs)}
        pstring = 1.0 * cirq.X.on(qs[1])
        pstring_pauli = from_cirq(pstring, len(qs))
        pstring_mpo = to_quimb(pstring_pauli, qs)
        psi = np.random.rand(2 ** nq).astype(complex)
        psi = psi / norm(psi)
        psi_mps = MatrixProductState.from_dense(psi)

        expectation_exact = pstring.expectation_from_state_vector(psi, qubit_map)
        expectation_mpo = (psi_mps.H @ psi_mps.gate_with_mpo(pstring_mpo)).real
        self.assertTrue(abs(expectation_exact - expectation_mpo) <= 1e-8)

    def test_three_qubits(self):
        nq = 3
        qs = cirq.LineQubit.range(nq)
        qubit_map = {q: i for i, q in enumerate(qs)}
        pstring = 1.0 * cirq.X.on(qs[0]) * cirq.Y.on(qs[2])
        pstring_pauli = from_cirq(pstring, len(qs))
        pstring_mpo = to_quimb(pstring_pauli, qs)
        psi = np.random.rand(2 ** nq).astype(complex)
        psi = psi / norm(psi)
        psi_mps = MatrixProductState.from_dense(psi)

        expectation_exact = pstring.expectation_from_state_vector(psi, qubit_map)
        expectation_mpo = (psi_mps.H @ psi_mps.gate_with_mpo(pstring_mpo)).real
        self.assertTrue(abs(expectation_exact - expectation_mpo) <= 1e-8)

if __name__ == "__main__":
    unittest.main()