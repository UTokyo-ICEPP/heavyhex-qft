"""Miscellaneous utility functions."""
import numpy as np
from scipy.sparse import coo_array
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
try:
    from qiskit_addon_sqd.qubit import matrix_elements_from_pauli
except ImportError:
    matrix_elements_from_pauli = None


def subspace_projection(op: SparsePauliOp, indices: np.ndarray) -> np.ndarray:
    if len(indices.shape) != 1:
        raise ValueError('Subspace projection indices must be a 1D array')

    if op.num_qubits < 32:
        return np.asarray(op.to_matrix(sparse=True)[indices[:, None], indices[None, :]].todense())

    if not matrix_elements_from_pauli:
        raise RuntimeError('Install qiskit-addon-sqd')

    shape = (op.num_qubits,) * 2
    mat = coo_array(shape, dtype=np.complex128)
    bitstring_matrix = ((indices[:, None] >> np.arange(op.num_qubits)[None, ::-1]) % 2).astype(bool)
    for pauli, coeff in zip(op.paulis, op.coeffs):
        data, row, col = matrix_elements_from_pauli(bitstring_matrix, pauli)
        mat += coo_array((coeff * data, (row, col)), shape)

    return mat


def as_bitarray(bitstr: str | np.ndarray):
    if isinstance(bitstr, str):
        return np.array(list(map(int, bitstr[::-1])))
    return np.asarray(bitstr)


def to_pauli_string(link_ops: dict[int, str], num_qubits: int) -> str:
    """Form the Pauli string corresponding to the given link operators."""
    pauli = ''
    for ilink, op in sorted(link_ops.items(), key=lambda x: x[0]):
        pauli = op.upper() + ('I' * (ilink - len(pauli))) + pauli
    return 'I' * (num_qubits - len(pauli)) + pauli


def qubit_coordinates(coupling_map: CouplingMap) -> list[list[int]]:
    """Compute the qubit coordinates that can be used with qiskit visualization functions.

    This function makes heavy-hex specific assumptions.
    """
    cgraph = coupling_map.graph.to_undirected()
    coords_r = []
    irow = -1
    icol = 0
    last_row_start = -1
    for iq in range(cgraph.num_nodes()):
        neighbors = cgraph.neighbors(iq)
        if iq + 1 in neighbors:
            if iq - 1 not in neighbors:
                last_row_start = iq
                irow += 1
                icol = 0
            coords_r.append([irow, icol])
            icol += 1
        elif iq - 1 in neighbors:
            coords_r.append([irow, icol])
            irow += 1
        elif min(neighbors) < iq:
            coords_r.append([irow, min(neighbors) - last_row_start])
        else:
            raise NotImplementedError('Chimney qubits!')

    nrows = irow + 1
    return [[nrows - irow - 1, icol] for irow, icol in coords_r]
