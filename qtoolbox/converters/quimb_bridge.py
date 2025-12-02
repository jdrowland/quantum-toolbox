from typing import List, Optional
import numpy as np
import cirq
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct
from qtoolbox.core.pauli import PauliString
from qtoolbox.converters.cirq_bridge import to_cirq

def pauli_string_to_mpo(pstring: cirq.PauliString, qs: List[cirq.Qid]) -> MatrixProductOperator:
    """Convert a Pauli string to a matrix product operator."""

    # Make a list of matrices for each operator in the string.
    ps_dense = pstring.dense(qs)
    matrices: List[np.ndarray] = []
    for pauli_int in ps_dense.pauli_mask:
        if pauli_int == 0:
            matrices.append(np.eye(2))
        elif pauli_int == 1:
            matrices.append(cirq.unitary(cirq.X))
        elif pauli_int == 2:
            matrices.append(cirq.unitary(cirq.Y))
        else: # pauli_int == 3
            matrices.append(cirq.unitary(cirq.Z))
    # Convert the matrices into tensors. We have a bond dim chi=1 for a Pauli string MPO.
    tensors: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if i == 0:
            if len(matrices) == 1:
                tensors.append(m)
            else:
                tensors.append(m.reshape((2, 2, 1)))
        elif i == len(matrices) - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return pstring.coefficient * MatrixProductOperator(tensors, shape="ludr")


def pauli_sum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int) -> MatrixProductOperator:
    """Convert a Pauli sum to an MPO."""

    if len(psum) == 0:
        raise ValueError("Paulisum passed has no terms.")

    for i, p in enumerate(psum):
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
    return mpo


def to_quimb(pauli: PauliString, qs: Optional[List[cirq.Qid]]=None) -> MatrixProductOperator:
    """Convert a PauliString to an MPO."""

    if qs is None:
        qs = cirq.LineQubit.range(pauli.n_qubits)
    pauli_cirq = to_cirq(pauli, qs)
    return pauli_string_to_mpo(pauli_cirq, qs)


if __name__ == "__main__":
    from qtoolbox.converters.cirq_bridge import from_cirq
    qs = cirq.LineQubit.range(3)
    ps_cirq = 1.0 * cirq.X.on(qs[0]) * cirq.Y.on(qs[1]) * cirq.Z.on(qs[2])
    ps = from_cirq(ps_cirq, len(qs))
    ps_mpo = to_quimb(ps)
    print(f"MPO has {len(ps_mpo.tensors)} tensors.")