# pylint: disable=no-member
"""Dual lattice of 2d Z2 LGT."""
from typing import Optional
import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from heavyhex_qft.pure_z2_lgt import PureZ2LGT, DummyPlaquette
from heavyhex_qft.utils import as_bitarray, to_pauli_string


class PlaquetteDual:
    r"""Dual lattice of 2d Z2 LGT.
    
    =========================
    "Encoding" and "decoding"
    =========================

    Plaquette dual of a Z2 LGT is closely related to classical error correction code. Specifically,
    we can regard the link configurations as code words that encode the messages represented by
    plaquette configurations. The "PL" matrix G

    .. math::

        G_{pl} = \begin{cases}
            1 \quad \text{if}\; l \in \partial p \\
            0 \quad \text{otherwise}
        \end{cases}

    is the generator of the code:

    .. math::

        x = y G + b

    where :math:`x` and :math:`y` are the link and plaquette configurations, and :math:`b` is the
    base link state.

    To decode a link configuration :math:`x` that belongs to the same charge sector as :math:`b`,
    we multiplly :math:`x + b` with the right inverse of :math:`G`:

    .. math::

        y = (x + b) G^{-1}

    The existence of a right inverse is intuitive but is formally guaranteed by the fact that the
    generator matrix of a linear code is full rank. It can thus be transformed to the standard form

    .. math::

        \tilde{G} = \left( I_{P} | -A \right) = V G S

    through column permutations :math:`S` and row operations :math:`V`. The right inverse of
    :math:`\tilde{G}` is trivially

    .. math::

        \tilde{G}^{-1} = \begin{pmatrix} I_{P} \\ 0 \end{pmatrix}.

    Then we have

    .. math::

        x = y V^{-1} \tilde{G} S^{-1} + b \\
        \therefore
        y = (x + b) S \tilde{G}^{-1} V.

    Explicit calculation of :math:`S` and :math:`V` is most easily done through a tracer matrix

    .. math::

        T = \begin{pmatrix} G & I_{P} \\
                            m & 0 \end{pmatrix},

    where :math:`m` is an index row vector. We apply the Gaussian elimination and column
    permutation that transform :math:`G` to :math:`\tilde{G}`. The row operation unitary will be
    recorded in the upper right block, and column permutations at the bottom row. After the
    transformation,

    .. math::

        \tilde{T} = \begin{pmatrix} \tilde{G} & V \\
                                    \pi(m) & 0 \end{pmatrix},

    where :math:`\pi` defines the permutation encoded by :math:`S` through

    .. math::

        S_{ij} = \delta_{j \pi(i)}.

    The :math:`j`th column of the permuted matrix has the value of the :math:`\pi^{-1}(j)`th
    column of the original. To decode a link state :math:`x`, then, we gather
    :math:`\pi^{-1}(m)`th elements of :math:`x + b` and multiply :math:`\tilde{G}^{-1} V`
    from the right.

    ============================
    Error-correction perspective
    ============================

    For any generator :math:`G` we can define a parity check matrix :math:`H` as a matrix whose
    right-multiplication kernel is spanned by the rows of :math:`G`:

    .. math::
    
        G H = 0.

    The parity check matrix is full rank but is not unique; any :math:`H' = HU` with (binary)
    unitary :math:`U` satisfies the above equation. A particular choice of such :math:`H` is then
    in the column echelon form, where the trailing entry of each column is 1, the trailing entry of
    every column is to the right of the trailing entry of every column to its left, and each row
    containing a trailing 1 has zeros in all its other entries. If we further perform row
    permutations to such :math:`H`, we can transform it to the standard-form parity check matrix

    .. math::

        \tilde{H} = \begin{pmatrix} A \\ I_{V} \end{pmatrix}.

    Let :math:`\tilde{H} = S^{-1} H U`. Then

    .. math::

        V G S S^{-1} H U = 0

    for any :math:`V`. Because :math:`\tilde{G} \tilde{H} = 0`, there must be a :math:`V` such that
    :math:`\tilde{G} = V G S^{-1}`.
    """
    def __init__(self, primal: PureZ2LGT, base_link_state: Optional[np.ndarray] = None):
        self.primal = primal
        if base_link_state is None:
            self.base_link_state = np.zeros(primal.num_links, dtype=np.uint8)
        else:
            assert len(base_link_state) == primal.num_links
            self.base_link_state = np.array(base_link_state, dtype=np.uint8)

        # Compute the row and column operations to transform the pl matrix to the standard form
        nrow = self.num_plaquettes
        ncol = self.num_links
        # Tracer matrix
        # - Top left block is the pl matrix to be transformed
        # - Top right block is the unitary representing the row operations
        # - Bottom left row traces the column permutations
        tracer = np.ones((nrow + 1, ncol + nrow), dtype=np.uint8)
        tracer[:nrow, :ncol] = self.primal.pl_matrix
        tracer[nrow, :ncol] = np.arange(ncol)
        tracer[:nrow, ncol:] = np.eye(nrow, dtype=np.uint8)

        for irow in range(nrow):
            one_hot = np.zeros(nrow, dtype=np.uint8)
            one_hot[irow] = 1
            # Search for a one-hot column with 1 at irow
            matches = np.argwhere(np.all(tracer[:nrow, :ncol] == one_hot[:, None], axis=0))
            if matches.shape[0] != 0:
                if (match := matches[0, 0]) != irow:
                    # col swap
                    tracer[:, [irow, match]] = tracer[:, [match, irow]]
                continue

            # Search for a row with 1 in column irow
            matches = np.argwhere(tracer[irow:nrow, irow] == 1)
            if matches.shape[0] != 0:
                if (match := matches[0, 0] + irow) != irow:
                    # row swap
                    tracer[[irow, match]] = tracer[[match, irow]]
            else:
                # Bring the column with 1 in row irow to column irow
                next_col = np.argwhere(tracer[irow, :ncol] == 1)[0, 0]
                tracer[:, [irow, next_col]] = tracer[:, [next_col, irow]]

            # Eliminate 1s everywhere except the current row
            mask = (tracer[:nrow, irow] == 1) & ~one_hot.astype(bool)
            tracer[:nrow][mask] ^= tracer[irow]

        self._decode_row_ops = np.array(tracer[:nrow, ncol:])
        self._decode_col_perms = np.array(tracer[nrow, :ncol])

    @property
    def graph(self) -> rx.PyGraph:
        return self.primal.dual_graph

    @property
    def num_plaquettes(self) -> int:
        return self.primal.num_plaquettes

    @property
    def num_active_plaquettes(self) -> int:
        return self.primal.num_active_plaquettes

    @property
    def num_links(self) -> int:
        return self.primal.num_links

    @property
    def plaquette_ids(self) -> list[int]:
        return self.primal.plaquette_ids

    @property
    def active_plaquette_ids(self) -> list[int]:
        return self.primal.active_plaquette_ids

    @property
    def link_ids(self) -> list[int]:
        return self.primal.link_ids

    def encode_plaq_to_link(self, plaq_state: np.ndarray | str) -> np.ndarray:
        plaq_state = as_bitarray(plaq_state)
        return (plaq_state @ self.primal.pl_matrix + self.base_link_state) & 1

    def decode_link_to_plaq(self, link_state: np.ndarray | str) -> np.ndarray:
        link_state = as_bitarray(link_state)
        return ((link_state ^ self.base_link_state)[self._decode_col_perms][:self.num_plaquettes]
                @ self._decode_row_ops) & 1

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Construct the Hamiltonian in the plaquette basis.

        The dual (Gauss's law-resolved) Hamiltonian encodes the charge sector of the base link state
        in the coefficients of the ZZ terms.
        """
        nq = self.num_active_plaquettes
        pid_to_iq = {plaq_id: iq for iq, plaq_id in enumerate(self.primal.active_plaquette_ids)}
        plaquettes = self.primal.graph.attrs['plaquettes']

        paulis = []
        coeffs = []
        for node1, node2, link_id in self.graph.edge_index_map().values():
            p1, p2 = self.graph[node1], self.graph[node2]
            ops = {}
            if isinstance(p1, int) and plaquettes[p1].active:
                ops[pid_to_iq[p1]] = 'Z'
            if isinstance(p2, int) and plaquettes[p2].active:
                ops[pid_to_iq[p2]] = 'Z'
            if ops:
                paulis.append(to_pauli_string(ops, nq))
                # Coeff is -1 / +1 if base link state is 0 / 1
                coeffs.append(-1. + 2. * self.base_link_state[self.primal._lid_to_idx[link_id]])
        paulis += [to_pauli_string({p: 'X'}, nq) for p in range(nq)]
        coeffs += [-plaquette_energy] * nq

        return SparsePauliOp(paulis, coeffs).simplify()

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        pid_to_iq = {plaq_id: iq for iq, plaq_id in enumerate(self.primal.active_plaquette_ids)}
        plaquettes = self.primal.graph.attrs['plaquettes']

        circuit = QuantumCircuit(self.num_active_plaquettes)
        for node1, node2, link_id in self.graph.edge_index_map().values():
            angle = (-1. + 2. * self.base_link_state[self.primal._lid_to_idx[link_id]]) * 2. * time
            p1, p2 = self.graph[node1], self.graph[node2]
            qubits = []
            if isinstance(p1, int) and plaquettes[p1].active:
                qubits.append(pid_to_iq[p1])
            if isinstance(p2, int) and plaquettes[p2].active:
                qubits.append(pid_to_iq[p2])
            if len(qubits) == 1:
                circuit.rz(angle, qubits[0])
            elif len(qubits) == 2:
                circuit.rzz(angle, *qubits)
        return circuit

    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.num_active_plaquettes)
        circuit.rx(-2. * plaquette_energy * time, range(self.num_active_plaquettes))
        return circuit
