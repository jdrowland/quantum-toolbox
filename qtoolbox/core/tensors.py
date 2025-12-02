"""Utilities for tensor networks."""

import numpy as np
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator

def mpo_mps_exepctation(mpo: MatrixProductOperator, mps: MatrixProductState) -> complex:
    """Get the expectation of an operator given the state.
    
    Arguments:
    mpo - Observable as an MPO.
    mps - State vector as an MPS.
    
    Returns:
    The expectation value."""

    mpo_times_mps = mpo.apply(mps)
    return mps.H @ mpo_times_mps


def mps_to_vector(mps: MatrixProductState) -> np.ndarray:
    """Convert an MPS into a normal vector. This assumes each index is a string
    followed by a number, e.g. three indices 'k0, k1, k2'.
    
    Arguments:
    mps - The mps to be converted to a vector."""

    def _idx_to_int(idx: str) -> int:
        digits = [c for c in idx if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        if len(digits) == 0:
            raise ValueError(f"Index {str} has no digits in it.")
        return int(''.join(digits))
    
    # Contract the MPS into a tensor, then sort the indices. Convert that to a vector.
    contracted_tensor = mps.contract()
    sorted_inds = sorted(contracted_tensor.inds, key=_idx_to_int)
    print(f"sorted_inds = {sorted_inds}")
    contracted_tensor.transpose(*sorted_inds, inplace=True)
    tensor_data = contracted_tensor.data
    return tensor_data.reshape((tensor_data.size,))