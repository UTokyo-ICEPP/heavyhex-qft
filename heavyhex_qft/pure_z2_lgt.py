# pylint: disable=no-member
"""Z2 lattice gauge theory with static charges."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import combinations, count
from typing import Optional
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import Backend
from qiskit_ibm_runtime.models import BackendProperties
from qiskit_ibm_runtime.models.exceptions import BackendPropertyError
from .utils import as_bitarray, to_pauli_string, qubit_coordinates


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

        # Cached vertex parity
        self._vertex_parity = None

    @property
    def num_plaquettes(self) -> int:
        return len(self.dual_graph.filter_nodes(lambda d: d is not None))

    @property
    def num_links(self) -> int:
        return self.graph.num_edges()

    @property
    def num_vertices(self) -> int:
        return self.graph.num_nodes()

    @property
    def num_qubits(self) -> int:
        return self.qubit_graph.num_nodes()

    def plaquette_dual(self, base_link_state: Optional[np.ndarray] = None) -> 'PlaquetteDual':
        return PlaquetteDual(self, base_link_state=base_link_state)

    def draw_graph(
        self,
        vertices: Optional[Sequence[int]] = None,
        links: Optional[Sequence[int]] = None,
        ax: Optional[Axes] = None
    ) -> Figure:
        kwargs = {'labels': str, 'edge_labels': str}
        if vertices is not None:
            kwargs['node_color'] = ['#1f78b4'] * self.num_vertices
            if len(vertices) == self.num_vertices:
                vertices = np.nonzero(vertices)[0]
            for iv in vertices:
                kwargs['node_color'][iv] = '#b41f1f'

        if links is not None:
            kwargs['edge_color'] = ['k'] * self.num_links
            if len(links) == self.num_links:
                links = np.nonzero(links)[0]
            for il in links:
                kwargs['edge_color'][il] = 'r'

        if (pos := self._graph_node_pos()) is not None:
            kwargs['pos'] = pos

        return rx.visualization.mpl_draw(self.graph, ax=ax, with_labels=True, **kwargs)

    def _graph_node_pos(self) -> dict[int, tuple[float, float]] | None:
        return None

    def draw_dual_graph(self, ax: Optional[Axes] = None) -> Figure:
        return rx.visualization.mpl_draw(self.dual_graph, ax=ax, with_labels=True, labels=str,
                                         edge_labels=str)

    def draw_qubit_graph(
        self,
        layout: Optional[list[int]] = None,
        coupling_map: Optional[Backend] = None,
        ax: Optional[Axes] = None
    ) -> Figure:
        kwargs = {
            'node_size': 200,
            'node_color': ['#cc8811'] * self.num_links + ['#88cc11'] * self.num_plaquettes
        }

        if layout is not None:
            def labels(payload):
                if payload[0] == 'link':
                    logical_qubit = payload[1]
                else:
                    logical_qubit = payload[1] + self.num_links
                return f'{layout[logical_qubit]}\n{payload[0][0]}:{payload[1]}'

            if coupling_map is not None:
                coords = qubit_coordinates(coupling_map)
                pos = {}
                for nidx in self.qubit_graph.node_indices():
                    payload = self.qubit_graph.get_node_data(nidx)
                    if payload[0] == 'link':
                        logical_qubit = payload[1]
                    else:
                        logical_qubit = payload[1] + self.num_links
                    row, col = coords[layout[logical_qubit]]
                    pos[nidx] = (col, row)

                kwargs['pos'] = pos
        else:
            labels = str

        kwargs['labels'] = labels

        return rx.visualization.mpl_draw(self.qubit_graph, ax=ax, with_labels=True, **kwargs)

    def plaquette_links(self, plaq_id: int) -> list[int]:
        """Return the list of ids of the links surrounding the plaquette."""
        return list(self.dual_graph.incident_edges(plaq_id))

    def link_plaquettes(self, link_id: int) -> list[int]:
        """Return the list (size 1 or 2) of ids of the plaquettes that have the link as an edge."""
        plaq_nodes = self.dual_graph.edge_index_map()[link_id][:2]
        return [plaq_id for plaq_id in plaq_nodes if self.dual_graph[plaq_id] is not None]

    def vertex_links(self, vertex_id: int) -> list[int]:
        """Return the list of ids of the links incident on the vertex.

        Note that the edge ids of the lattice graph and the node ids of the corresponding link
        qubits in the coincident.
        """
        return list(self.graph.incident_edges(vertex_id))

    def link_vertices(self, link_id: int) -> tuple[int, int]:
        """Return the ids of the pair of vertices that the link connects."""
        return self.graph.edge_index_map()[link_id][:2]

    def layout_heavy_hex(
        self,
        coupling_map: CouplingMap,
        qubit_assignment: Optional[int | dict[tuple[str, int], int]] = None,
        backend_properties: Optional[BackendProperties] = None,
        basis_2q: str = 'cx'
    ) -> list[int]:
        """Return the physical qubit layout of the qubit graph using qubits in the coupling map.

        If qubit_assignment is not given or results in multiple layout candidates, the candidate
        with the smallest product of gate and readout errors are used. If backend properties are
        also not given, a candidate is randomly chosen.

        Args:
            coupling_map: backend.coupling_map.
            qubit_assignment: Physical qubit id to assign link 0 to, or an assignment hint dict of
                form {('link' or 'plaq', id): (physical qubit)}.
            backend_properties: Backend properties to extract the gate and readout error data from.
            basis_2q: 2-qubit gate specification for the magnetic circuit.

        Returns:
            List of physical qubit ids to be passed to the transpiler.
        """
        cgraph = coupling_map.graph.to_undirected()
        for idx in cgraph.node_indices():
            cgraph[idx] = (idx, tuple(cgraph.neighbors(idx)))

        if qubit_assignment is None:
            node_matcher = None
        else:
            if isinstance(qubit_assignment, int):
                qubit_assignment = {('link', 0): qubit_assignment}

            def node_matcher(physical_qubit_data, lattice_qubit_data):
                physical_qubit, physical_neighbors = physical_qubit_data
                node_type, obj_id = lattice_qubit_data
                # True if this is an assigned qubit
                if (assignment := qubit_assignment.get(lattice_qubit_data)) is not None:
                    return assignment == physical_qubit
                # Otherwise recall class-default matcher
                return self._layout_node_matcher(physical_qubit, physical_neighbors, node_type,
                                                 obj_id)

        mappings = rx.vf2_mapping(cgraph, self.qubit_graph, node_matcher=node_matcher,
                                  subgraph=True, induced=False)
        mappings = list(mappings)
        if len(mappings) == 0:
            raise ValueError('Layout with the given qubit assignment could not be found.')

        # 2Q basis gate for the backend
        twoq_gate_name = ''
        if backend_properties is not None:
            twoq_gate_name = next(gate_prop.gate for gate_prop in backend_properties.gates
                                  if gate_prop.gate in ['ecr', 'cz'])

        score_max, best_layout = None, None
        for mapping in mappings:
            layout = [None] * self.qubit_graph.num_nodes()
            for physical_qubit, logical_qubit in mapping.items():
                layout[logical_qubit] = physical_qubit

            if backend_properties is None:
                best_layout = layout
                break

            # Readout errors of the link qubits
            log_error_score = sum(
                np.log(1. - min(backend_properties.readout_error(qubit), 0.99999))
                for qubit in layout[:self.num_links]
            )
            # 2Q gate errors
            for (gate, logical_qubits), counts in self.magnetic_2q_gate_counts(basis_2q).items():
                if gate in ['cx', 'cz']:
                    gate_name = twoq_gate_name
                else:
                    gate_name = gate  # rzz
                qubits = tuple(layout[qubit] for qubit in logical_qubits)
                try:
                    gate_error = backend_properties.gate_error(gate_name, qubits)
                except BackendPropertyError:
                    gate_error = backend_properties.gate_error(gate_name, qubits[::-1])
                error = min(gate_error, 0.99999)
                log_error_score += np.log(1. - error) * counts
            # If best score, remember the layout
            if score_max is None or log_error_score > score_max:
                score_max = log_error_score
                best_layout = layout

        if best_layout is None:
            raise ValueError('I do not think this would ever happen')

        return best_layout

    @abstractmethod
    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        node_type: str,
        obj_id: int
    ) -> bool:
        """Node matcher function for qubit mapping."""

    def get_syndrome(self, link_state: np.ndarray | str) -> np.ndarray:
        """Compute the bit-flip syndrome (parity of sum of link 0/1s at each vertex) from a link
        measurement result."""
        link_state = as_bitarray(link_state)
        return np.array([np.sum(link_state[self.vertex_links(iv)]) % 2
                         for iv in range(self.num_vertices)])

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Return the Z2 LGT Hamiltonian expressed as a SparsePauliOp.

        The lengths of the Pauli strings equal the number of links in the lattice, not the number
        of qubits.
        """
        link_terms = [to_pauli_string({lid: 'Z'}, self.num_links)
                      for lid in self.graph.edge_indices()]
        plaquette_terms = [to_pauli_string({lid: 'X' for lid in self.plaquette_links(plid)},
                                           self.num_links)
                           for plid in range(self.num_plaquettes)]
        hamiltonian = SparsePauliOp(link_terms, [1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [plaquette_energy] * len(plaquette_terms))
        return hamiltonian

    def charge_subspace(self, vertex_charge: list[int]) -> np.ndarray:
        """Return the dimensions of the full Hilbert space (d=2**num_link) that span the subspace
        of the given vertex charges.
        """
        if self.num_links > 64:
            raise NotImplementedError('charge_subspace method is only available for lattices with'
                                      ' <= 64 links')
        if len(vertex_charge) != self.num_vertices or any(c not in (0, 1) for c in vertex_charge):
            raise ValueError(f'Argument must be a length-{self.num_vertices} list of 0 or 1')

        # Bit patterns of link excitations - final shape will be (subspace_dim, num_links)
        mask_all = np.array([[False] * self.num_links])
        covered_lids = set()
        for ivert, parity in enumerate(vertex_charge):
            lids = self.vertex_links(ivert)
            nl = len(lids)

            # Bit patterns of link excitations for this vertex
            # The number of combinations is [sum_{j in even or odd} nCj] where n is the number of
            # links. Because (1 - 1)^n = sum_{j}(-1)^j*nCj = 0 and (1 + 1)^n = sum_{j}nCj = 2^n, we
            # know that we have 2^{n-1} combinations.
            mask_vertex = np.zeros((2 ** (nl - 1), self.num_links), dtype=bool)
            icomb = iter(count())
            # Loop over all combinations of link excitations that produce the given parity
            for nup in range(parity, nl + 1, 2):
                for comb in combinations(lids, nup):
                    mask_vertex[next(icomb), list(comb)] = True

            # Will compare the mask for vertex with the aggregate mask only over overlapping links
            compared_lids = list(set(lids) & set(covered_lids))
            # Links must be either both excited or unexcited to be compatible
            compatible = np.logical_not(
                np.any(
                    np.logical_xor(
                        mask_vertex[None, :, compared_lids],
                        mask_all[:, None, compared_lids]
                    ),
                    axis=2
                )
            )
            # Update the aggregate mask
            mask_all = (mask_vertex[None, ...] | mask_all[:, None, :])[compatible]
            covered_lids.update(lids)

        indices = np.sum(
            np.asarray(mask_all, dtype=np.uint64) << np.arange(self.num_links, dtype=np.uint64),
            axis=1
        )
        return np.sort(indices)

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        circuit.rz(-2. * time, range(self.num_links))
        return circuit

    @abstractmethod
    def magnetic_evolution(
        self,
        plaquette_energy: float,
        time: float,
        basis_2q: str = 'cx'
    ) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""

    @abstractmethod
    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cx'
    ) -> dict[tuple[str, tuple[int, int]], int]:
        """Return a list of (gate name, qubits, counts)."""


class PlaquetteDual:
    """Dual lattice of 2d Z2 LGT."""
    def __init__(self, primal: PureZ2LGT, base_link_state: Optional[np.ndarray] = None):
        self._primal = primal
        if base_link_state is None:
            self._base_link_state = np.zeros(primal.num_links)
        else:
            self._base_link_state = np.array(base_link_state)

    def map_link_state(self, link_state: np.ndarray | str) -> np.ndarray:
        link_state = as_bitarray(link_state)
        if np.max(link_state) > 1 or np.min(link_state) < 0:
            raise ValueError('Non-binary link state')
        if link_state.shape[0] != self._base_link_state.shape[0]:
            raise ValueError('Number of links inconsistent with the primal graph')

        # Excited links are those whose states differ from the base
        excited_links = np.nonzero(link_state != self._base_link_state)[0]

        # Cut the dual graph along excited links
        dual_graph = self._primal.dual_graph.copy()
        for link in excited_links:
            dual_graph.remove_edge_from_index(link)
        # Remove boundary links too
        for link, edge_info in dual_graph.edge_index_map().items():
            if None in [dual_graph[node] for node in edge_info[:2]]:
                dual_graph.remove_edge_from_index(link)
        patches = rx.connected_components(dual_graph)

        # Next: Determine which patches are excited.
        # Construct a new graph whose vertices are the patches
        patch_graph = self._primal.dual_graph.copy()
        edge_index_map = patch_graph.edge_index_map()
        # Relabel boundary links
        for link in excited_links:
            if None in [patch_graph[node] for node in edge_index_map[link][:2]]:
                patch_graph.update_edge_by_index(link, -edge_index_map[link][2] - 1)

        for plaquettes in patches:
            plaquettes = list(plaquettes)
            if len(plaquettes) == 1 and patch_graph[plaquettes[0]] is None:
                continue
            patch_graph.contract_nodes(plaquettes, plaquettes)

        # Apply a two-coloring and determine which of 0 or 1 corresponds to excitation
        coloring = rx.two_color(patch_graph)
        # Find a boundary plaquette
        edge_index_map = patch_graph.edge_index_map()
        p1, p2, weight = next(edge_info for edge_info in edge_index_map.values()
                              if None in [patch_graph[node] for node in edge_info[:2]])
        boundary = next(patch for patch in [p1, p2] if patch_graph[patch] is not None)
        if weight > 0:
            base_color = coloring[boundary]
        else:
            base_color = 1 - coloring[boundary]

        # Finally compose the plaquette state
        state = np.zeros(self._primal.num_plaquettes, dtype=int)
        for patch in patch_graph.node_indices():
            if coloring[patch] != base_color and (plaquettes := patch_graph[patch]) is not None:
                state[plaquettes] = 1

        return state

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Construct the Hamiltonian in the plaquette basis."""
        num_p = self._primal.num_plaquettes
        paulis = []
        for p1, p2, _ in self._primal.dual_graph.edge_index_map().values():
            if self._primal.dual_graph[p1] is None:
                paulis.append(to_pauli_string({p2: 'Z'}, num_p))
            elif self._primal.dual_graph[p2] is None:
                paulis.append(to_pauli_string({p1: 'Z'}, num_p))
            else:
                paulis.append(to_pauli_string({p1: 'Z', p2: 'Z'}, num_p))
        coeffs = [1.] * len(paulis)
        paulis += [to_pauli_string({p: 'X'}, num_p) for p in range(num_p)]
        coeffs += [plaquette_energy] * num_p

        return SparsePauliOp(paulis, coeffs)
