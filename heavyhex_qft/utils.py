"""Miscellaneous utility functions."""
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def subspace_projection(op: SparsePauliOp, indices: np.ndarray) -> np.ndarray:
    if len(indices.shape) != 1:
        raise ValueError('Subspace projection indices must be a 1D array')

    return np.asarray(op.to_matrix(sparse=True)[indices[:, None], indices[None, :]].todense())
