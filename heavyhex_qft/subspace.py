"""Subspace projection routines."""
from functools import partial
import numpy as np
from scipy.sparse import coo_array
import jax
import jax.numpy as jnp
from qiskit.quantum_info import SparsePauliOp

PAULI_BIT_MAP = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])
PAULI_COEFFS = np.array([
    [1., 1.],
    [1., 1.],
    [1.j, -1.j],
    [1., -1.]
])
PAULI_INDEX = {c: i for i, c in enumerate('IXYZ')}


def subspace_projection(op: SparsePauliOp, indices: np.ndarray) -> np.ndarray:
    if len(indices.shape) == 1:
        if op.num_qubits < 32:
            return np.asarray(
                op.to_matrix(sparse=True)[indices[:, None], indices[None, :]].todense()
            )
        indices = (indices[:, None] >> np.arange(op.num_qubits)[None, ::-1]) % 2

    col_bitstrings = jnp.array(indices)
    paulistrings = jnp.array([[PAULI_INDEX[c] for c in p.tolabel()] for p in op.paulis])
    row_bitstrings, coeffs = map_bitstrings_by_paulis(col_bitstrings, paulistrings)

    # filter out row_bitstrings with indices and construct a coo array


def map_bitstring_by_pauli(bitstring: np.ndarray, paulistring: np.ndarray, npmod=np):
    mapped_bitstring = PAULI_BIT_MAP[paulistring, bitstring]
    coeff = npmod.prod(PAULI_COEFFS[paulistring, bitstring])
    return mapped_bitstring, coeff


jmap_bitstring_by_pauli = jax.jit(partial(map_bitstring_by_pauli, npmod=jnp))
map_bitstrings_by_pauli = jax.jit(jax.vmap(jmap_bitstring_by_pauli, in_axes=(0, None)))
map_bitstrings_by_paulis = jax.jit(jax.vmap(map_bitstrings_by_pauli, in_axes=(None, 0)))


@jax.jit
def bitstring_isin(bitstring: jax.Array, pool: jax.Array):
    def fun(carry, pos):
        low, high = carry
        sublow = jnp.searchsorted(bitstring[pos], pool[low:high, pos], 'left')
        subhigh = jnp.searchsorted(bitstring[pos], pool[low:high, pos], 'right')
        high = low + subhigh
        low += sublow
        return (low, high), jnp.greater(high, low)

    num_qubits = bitstring.shape[0]
    (low, high), matches = jax.lax.scan(fun, (0, pool.shape[0]), jnp.arange(num_qubits))
    return jax.lax.cond(matches[-1], low, -1)
