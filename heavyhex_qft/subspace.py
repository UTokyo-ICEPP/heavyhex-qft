"""Subspace projection routines."""
from functools import partial
import numpy as np
# from scipy.sparse import coo_array
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, bcoo_dot_general
from qiskit.quantum_info import SparsePauliOp
from .linalg import lobpcg_standard

PAULI_BIT_MAP = jnp.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
], dtype=np.uint8)
PAULI_COEFFS = jnp.array([
    [1., 1.],
    [1., 1.],
    [1.j, -1.j],
    [1., -1.]
])
PAULI_INDEX = {c: i for i, c in enumerate('IXYZ')}


def subspace_projection(op: SparsePauliOp, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices)
    if len(indices.shape) == 1:
        if op.num_qubits < 32:
            return np.asarray(
                op.to_matrix(sparse=True)[indices[:, None], indices[None, :]].todense()
            )
        indices = (indices[:, None] >> np.arange(op.num_qubits)[None, ::-1]) % 2

    proj_dim = indices.shape[0]
    indices = jnp.array(indices)
    paulistrings = jnp.array([[PAULI_INDEX[c] for c in p.to_label()] for p in op.paulis])
    mapped_bitstrings, pauli_mes = map_bitstrings_by_paulis(indices, paulistrings)
    rows = bitstring_positions(mapped_bitstrings.reshape((-1, op.num_qubits)), indices)
    coeffs = (pauli_mes * op.coeffs[:, None]).reshape(-1)
    in_subspace = jnp.not_equal(rows, -1)
    data = coeffs[in_subspace]
    rows = rows[in_subspace]
    cols = jnp.tile(jnp.arange(proj_dim), len(op.paulis))[in_subspace]
    indices = jnp.stack([rows, cols], axis=1)
    return BCOO((data, indices), shape=(proj_dim, proj_dim))


def map_bitstring_by_pauli(bitstring: np.ndarray, paulistring: np.ndarray, npmod=np):
    mapped_bitstring = PAULI_BIT_MAP[paulistring, bitstring]
    coeff = npmod.prod(PAULI_COEFFS[paulistring, bitstring])
    return mapped_bitstring, coeff


jmap_bitstring_by_pauli = jax.jit(partial(map_bitstring_by_pauli, npmod=jnp))
map_bitstrings_by_pauli = jax.jit(jax.vmap(jmap_bitstring_by_pauli, in_axes=(0, None)))
map_bitstrings_by_paulis = jax.jit(jax.vmap(map_bitstrings_by_pauli, in_axes=(None, 0)))


@jax.jit
def bitstring_position(bitstring: jax.Array, pool: jax.Array):
    matches = jnp.all(jnp.equal(bitstring[None, :], pool), axis=1)
    idx = jnp.argmax(matches)
    return jax.lax.select(jnp.equal(idx, 0), jax.lax.select(matches[0], 0, -1), idx)


bitstring_positions = jax.jit(jax.vmap(bitstring_position, in_axes=(0, None)))


def multiply_bcoo(x, bcoo):
    return bcoo_dot_general(-bcoo, x, dimension_numbers=(([1], [0]), ([], [])))


def subspace_ground_state(mat):
    xmat = jnp.ones((mat.shape[0], 1), dtype=np.complex128)
    # pylint: disable-next=unbalanced-tuple-unpacking
    jeigvals, jeigvecs, _ = lobpcg_standard(multiply_bcoo, xmat, args=(mat,))
    return float(-jeigvals[0]), np.array(jeigvecs[:, 0])
