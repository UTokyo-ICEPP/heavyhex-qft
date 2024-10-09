"""Z2 lattice gauge theory with static charges."""
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from matplotlib.figure import Figure
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp


class PureZ2LGT(ABC):
    r"""Base class for Z2 lattice gauge theory with static charges (gauge dof only).

    Hamiltonian of the theory is

    .. math::

        H = -\sum_{e \in \mathcal{E}} Z_e - K \sum_{p \in \mathcal{P}} \prod_{e \in \partial p} X_e,

    where :math:`\mathcal{E}` is the set of all edges and :math:`\mathcal{P}` is the set of all
    plaquettes, with :math:`\partial p` corresponding to the edges of the plaquette :math:`p`.

    Subclasses should construct two undirected graphs from their specific topologies. The first
    graph represents the physical lattice where the nodes and edges correspond to matter sites
    (vertices) and links, respectively. The second graph gives the qubit implementation of the
    lattice, where the nodes correspond to qubits and can represent links or plaquettes, and the
    edges to the connections between qubits. The nodes and edges of the physical graph are labeled
    with integers corresponding to vertex and link ids. The nodes of the qubit graph are instead
    labeled by 2-tuples with form (type, id) where type is either 'link', 'anc', or 'plaq', and id
    is the serial number of the edge within each type (link id must coincide with the edge ids of
    the physical graph). For a lattice with L links and P plaquettes, the qubit graph with A
    ancillas should have L+A+P nodes, with the first L representing the links, next A representing
    the ancillas, and the last P as the plaquettes.
    """
    def __init__(self, num_vertices: int):
        self.graph = rx.PyGraph()
        self.graph.add_nodes_from(range(num_vertices))
        self.dual_graph = rx.PyGraph()
        self.qubit_graph = rx.PyGraph()

    @property
    def num_plaquettes(self) -> int:
        return self.dual_graph.num_nodes()

    @property
    def num_links(self) -> int:
        return self.graph.num_edges()

    @property
    def num_vertices(self) -> int:
        return self.graph.num_nodes()

    def draw_graph(self) -> Figure:
        return rx.visualization.mpl_draw(self.graph, with_labels=True, labels=str, edge_labels=str)

    def draw_dual_graph(self) -> Figure:
        return rx.visualization.mpl_draw(self.dual_graph, with_labels=True, labels=str,
                                         edge_labels=str)

    def draw_qubit_graph(self) -> Figure:
        return rx.visualization.mpl_draw(self.qubit_graph, with_labels=True, labels=str)

    def plaquette_links(self, plaq_id: int) -> list[int]:
        """Return the list of ids of the links surrounding the plaquette."""
        return list(self.dual_graph.incident_edges(plaq_id))

    def vertex_links(self, vertex_id: int) -> list[int]:
        """Return the list of ids of the links incident on the vertex.

        Note that the edge ids of the lattice graph and the node ids of the corresponding link
        qubits in the coincident.
        """
        return list(self.graph.incident_edges(vertex_id))

    def layout_heavy_hex(
        self,
        coupling_map: CouplingMap,
        qubit_assignment: Union[int, dict[tuple[str, int], int]]
    ) -> list[int]:
        """Return the physical qubit layout of the qubit graph using qubits in the coupling map.

        Args:
            coupling_map: backend.coupling_map.
            qubit_assignment: Physical qubit id to assign link 0 to, or an assignment hint dict of
                form {('link' or 'plaq', id): (physical qubit)}.

        Returns:
            List of physical qubit ids to be passed to the transpiler.
        """
        cgraph = coupling_map.graph.to_undirected()
        for idx in cgraph.node_indices():
            cgraph[idx] = (idx, tuple(cgraph.neighbors(idx)))

        if isinstance(qubit_assignment, int):
            qubit_assignment = {('link', 0): qubit_assignment}

        def node_matcher(physical_qubit_data, lattice_qubit_data):
            physical_qubit, physical_neighbors = physical_qubit_data
            node_type, obj_id = lattice_qubit_data
            # True if this is an assigned qubit
            if qubit_assignment.get(lattice_qubit_data) == physical_qubit:
                return True
            # Otherwise recall specific matcher
            return self._layout_node_matcher(physical_qubit, physical_neighbors, node_type, obj_id)

        vf2 = rx.vf2_mapping(cgraph, self.qubit_graph, node_matcher=node_matcher, subgraph=True,
                             induced=False)
        try:
            mapping = next(vf2)
        except StopIteration as exc:
            raise ValueError('Layout with the given qubit assignment could not be found.') from exc

        layout = [None] * self.qubit_graph.num_nodes()
        for physical_qubit, logical_qubit in mapping.items():
            layout[logical_qubit] = physical_qubit

        return layout

    @abstractmethod
    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        node_type: str,
        obj_id: int
    ) -> bool:
        """Node matcher function for qubit mapping."""

    def to_pauli(self, link_ops: dict[int, str], pad_plaquettes: bool = False) -> str:
        """Form the Pauli string corresponding to the given link operators.

        If pad_plaquettes is True, ('I' * num_plaquettes) is appended to the returned string.
        """
        link_paulis = []
        for link_id, op in link_ops.items():
            link_paulis.append((link_id, op.upper()))

        link_paulis = sorted(link_paulis, key=lambda x: x[0])
        pauli = ''
        for link, op in link_paulis:
            pauli = op + ('I' * (link - len(pauli))) + pauli

        pauli = 'I' * (self.num_links - len(pauli)) + pauli
        if pad_plaquettes:
            pauli = 'I' * self.num_plaquettes + pauli
        return pauli

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Return the Z2 LGT Hamiltonian expressed as a SparsePauliOp.

        The lengths of the Pauli strings equal the number of links in the lattice, not the number
        of qubits.
        """
        link_terms = [self.to_pauli({lid: 'Z'}) for lid in self.graph.edge_indices()]
        plaquette_terms = [self.to_pauli({lid: 'X' for lid in self.plaquette_links(plid)})
                           for plid in self.dual_graph.node_indices()]
        hamiltonian = SparsePauliOp(link_terms, [-1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [-plaquette_energy] * len(plaquette_terms))
        return hamiltonian

    def charge_subspace(self, vertex_charge: list[int]) -> np.ndarray:
        """Return the dimensions of the full Hilbert space (d=2**num_link) that span the subspace
        of the given vertex charges.

        TODO This implementation is very likely not efficient and unnecessarily restricts the
        usability of the method to smaller number of links.
        """
        if len(vertex_charge) != self.num_vertices or any(c not in (0, 1) for c in vertex_charge):
            raise ValueError(f'Argument must be a length-{self.num_vertices} list of 0 or 1')

        hspace_dim = 2 ** self.num_links
        basis_indices = np.arange(hspace_dim)
        all_bitstrings = np.array([(basis_indices >> i) % 2
                                   for i in range(self.num_links)], dtype='uint8').T
        flt = np.ones(hspace_dim, dtype='uint8')
        for vertex, parity in enumerate(vertex_charge):
            links = self.vertex_links(vertex)
            flt *= np.asarray(np.sum(all_bitstrings[:, links], axis=1) % 2 == parity, dtype='uint8')

        return np.nonzero(flt)[0]

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        circuit.rz(-2. * time, range(self.num_links))
        return circuit

    @abstractmethod
    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
