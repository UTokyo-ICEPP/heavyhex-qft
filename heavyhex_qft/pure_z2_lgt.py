# pylint: disable=no-member
"""Z2 lattice gauge theory with static charges."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations, count
from typing import Any, Optional
import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.models import BackendProperties
from .utils import as_bitarray, to_pauli_string, qubit_coordinates

LOG = logging.getLogger(__name__)


@dataclass
class Vertex:
    """Vertex data."""
    vertex_id: int
    position: tuple[float, float]
    plaquettes: set[int] = field(default_factory=set)


@dataclass
class Link:
    """Link data."""
    link_id: int
    logical_qubit: int = -1


@dataclass
class Plaquette:
    """Plaquette data."""
    plaq_id: int
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)
    logical_qubit: int | None = None
    direct_link: Link | None = None


@dataclass
class DummyPlaquette:
    """Dummy plaquette for dual graph."""
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)


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
    def __init__(self, graph: rx.PyGraph, dual_graph: rx.PyGraph, qubit_graph: rx.PyGraph):
        self.graph = graph
        self.dual_graph = dual_graph
        self.qubit_graph = qubit_graph

        # Cached vertex parity
        self._vertex_parity = None

    @property
    def num_plaquettes(self) -> int:
        return len(self.dual_graph.filter_nodes(lambda p: isinstance(p, Plaquette)))

    @property
    def num_links(self) -> int:
        return self.graph.num_edges()

    @property
    def num_vertices(self) -> int:
        return self.graph.num_nodes()

    @property
    def num_qubits(self) -> int:
        return self.qubit_graph.num_nodes()

    def plaquette_dual(self, base_link_state: Optional[np.ndarray] = None):
        # pylint: disable-next=import-outside-toplevel
        from .plaquette_dual import PlaquetteDual
        return PlaquetteDual(self, base_link_state=base_link_state)

    def draw_graph(
        self,
        vertices: Optional[Sequence[int]] = None,
        links: Optional[Sequence[int]] = None,
        ax: Optional[Axes] = None
    ) -> Figure:
        kwargs = {'labels': str, 'edge_labels': str,
                  'pos': {vertex.vertex_id: vertex.position for vertex in self.graph.nodes()}}
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

        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.graph.num_nodes()))
        for link_id, (n1, n2, _) in self.graph.edge_index_map().items():
            graph.add_edge(n1, n2, link_id)

        fig = rx.visualization.mpl_draw(graph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()

        # Draw the plaquette ids
        for plaquette in self.dual_graph.nodes():
            if isinstance(plaquette, DummyPlaquette):
                continue
            x, y = plaquette.position
            fig.axes[0].text(x, y, f'{plaquette.plaq_id}', ha='center', va='center')

        if not plt.isinteractive() or ax is None:
            return fig

    def draw_dual_graph(self, ax: Optional[Axes] = None) -> Figure:
        kwargs = {'labels': str, 'edge_labels': str,
                  'pos': {nid: self.dual_graph[nid].position
                          for nid in self.dual_graph.node_indices()}}

        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.num_plaquettes))
        graph.add_nodes_from([''] * (self.dual_graph.num_nodes() - self.num_plaquettes))
        for link_id, (n1, n2, _) in self.dual_graph.edge_index_map().items():
            graph.add_edge(n1, n2, link_id)

        fig = rx.visualization.mpl_draw(graph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()

        # Draw the vertex ids
        for vertex in self.graph.nodes():
            x, y = vertex.position
            fig.axes[0].text(x, y, f'{vertex.vertex_id}', ha='center', va='center')

        if not plt.isinteractive() or ax is None:
            return fig

    def draw_qubit_graph(
        self,
        layout: list[int],
        coupling_map: CouplingMap,
        ax: Optional[Axes] = None,
        links: Optional[Sequence[int]] = None,
        plaquettes: Optional[Sequence[int]] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        **kwargs
    ) -> Figure:
        cgraph = coupling_map.graph.to_undirected()
        physical_graph = rx.PyGraph(multigraph=False, node_count_hint=cgraph.num_nodes())
        physical_graph.add_nodes_from(cgraph.node_indices())
        for source, target in cgraph.edge_list():
            physical_graph.add_edge(source, target, None)

        selected_links = set(links or [])
        selected_plaquettes = set(plaquettes or [])
        selected_qubits = set(physical_qubits or [])

        node_color = [None] * physical_graph.num_nodes()
        for qobj in self.qubit_graph.nodes():
            physical_qubit = layout[qobj.logical_qubit]
            if isinstance(qobj, Link):
                physical_graph[physical_qubit] = f'{physical_qubit}\nL:{qobj.link_id}'
                if qobj.link_id in selected_links:
                    node_color[physical_qubit] = '#ffaaff'
                else:
                    node_color[physical_qubit] = '#cc11cc'
            elif isinstance(qobj, Plaquette):
                physical_graph[physical_qubit] = f'{physical_qubit}\nP:{qobj.plaq_id}'
                if qobj.plaq_id in selected_plaquettes:
                    node_color[physical_qubit] = '#aaffff'
                else:
                    node_color[physical_qubit] = '#11cccc'

        for physical_qubit in set(coupling_map.physical_qubits) - set(layout):
            physical_graph[physical_qubit] = f'{physical_qubit}'
            if physical_qubit in selected_qubits:
                node_color[physical_qubit] = '#cccccc'
            else:
                node_color[physical_qubit] = '#888888'

        pos = {iq: (col, row) for iq, (row, col) in enumerate(qubit_coordinates(coupling_map))}

        kwargs['pos'] = pos
        if 'node_size' not in kwargs:
            kwargs['node_size'] = 160
        if 'node_color' not in kwargs:
            kwargs['node_color'] = node_color
        if 'font_size' not in kwargs:
            kwargs['font_size'] = 8

        fig = rx.visualization.mpl_draw(physical_graph, ax=ax, with_labels=True, labels=str,
                                        **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()
        # Add link drawings
        self._draw_qubit_graph_links(layout, pos, selected_links, ax or fig.axes[0])

        if not plt.isinteractive() or ax is None:
            return fig

    def _draw_qubit_graph_links(self, layout, pos, selected_links, ax):
        """Draw links on the qubit graph plot."""

    def plaquette_links(self, plaq_id: int) -> list[int]:
        """Return the list of ids of the links surrounding the plaquette."""
        return list(self.dual_graph.incident_edges(plaq_id))

    def link_plaquettes(self, link_id: int) -> list[int]:
        """Return the list (size 1 or 2) of ids of the plaquettes that have the link as an edge."""
        nids = self.dual_graph.edge_index_map()[link_id][:2]
        return [nid for nid in nids if isinstance(self.dual_graph[nid], Plaquette)]

    def vertex_links(self, vertex_id: int) -> list[int]:
        """Return the list of ids of the links incident on the vertex."""
        return list(self.graph.incident_edges(vertex_id))

    def link_vertices(self, link_id: int) -> tuple[int, int]:
        """Return the ids of the pair of vertices that the link connects."""
        return self.graph.edge_index_map()[link_id][:2]

    def layout_heavy_hex(
        self,
        coupling_map: Optional[CouplingMap] = None,
        qubit_assignment: Optional[int | dict[tuple[str, int], int]] = None,
        backend_properties: Optional[BackendProperties] = None,
        target: Optional[Target] = None,
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
            target: Backend target. If given, overrides coupling_map and backend_properties.
            basis_2q: 2-qubit gate specification for the magnetic circuit.

        Returns:
            List of physical qubit ids to be passed to the transpiler.
        """
        gate_errors = {}
        if target:
            coupling_map = target.build_coupling_map()
            readout_errors = [target['measure'][(iq,)].error for iq in range(target.num_qubits)]
            for inst, qargs in target.instructions:
                if inst.num_qubits == 2:
                    gate_errors[(inst.name, qargs)] = target[inst.name][qargs].error

        elif backend_properties:
            readout_errors = [backend_properties.readout_error(iq)
                              for iq in range(len(backend_properties.qubits))]
            for gate_prop in backend_properties.gates:
                if len(gate_prop.qubits) == 2:
                    try:
                        error = next(param.value for param in gate_prop.parameters
                                     if param.name == 'gate_error')
                    except StopIteration:
                        error = 0.99999
                    gate_errors[(gate_prop.gate, tuple(gate_prop.qubits))] = error

        cgraph = coupling_map.graph.to_undirected()
        for idx in cgraph.node_indices():
            cgraph[idx] = (idx, tuple(cgraph.neighbors(idx)))

        if qubit_assignment is None:
            LOG.info('[layout_heavy_hex] qubit_assignment not given. Using all candidates.')
            node_matcher = None
        else:
            if isinstance(qubit_assignment, int):
                qubit_assignment = {('link', 0): qubit_assignment}

            LOG.info('[layout_heavy_hex] Qubit assignment: %s', qubit_assignment)

            def node_matcher(physical_qubit_data, lattice_qubit_data):
                physical_qubit, physical_neighbors = physical_qubit_data
                qobj = lattice_qubit_data
                # True if this is an assigned qubit
                if isinstance(qobj, Link):
                    obj_type = 'link'
                    obj_id = qobj.link_id
                else:
                    obj_type = 'plaq'
                    obj_id = qobj.plaq_id
                if (assignment := qubit_assignment.get((obj_type, obj_id))) is not None:
                    return assignment == physical_qubit
                # Otherwise recall class-default matcher
                return self._layout_node_matcher(physical_qubit, physical_neighbors, qobj)

        LOG.info('[layout_heavy_hex] Finding layout candidates with vf2_mapping..')
        mappings = rx.vf2_mapping(cgraph, self.qubit_graph, node_matcher=node_matcher,
                                  subgraph=True, induced=False)
        LOG.info('[layout_heavy_hex] Done.')
        mappings = list(mappings)
        if len(mappings) == 0:
            raise ValueError('Layout with the given qubit assignment could not be found.')

        score_max, best_layout = None, None
        for mapping in mappings:
            layout = [None] * self.qubit_graph.num_nodes()
            for physical_qubit, logical_qubit in mapping.items():
                layout[logical_qubit] = physical_qubit

            if not gate_errors:
                best_layout = layout
                LOG.info('[layout_heavy_hex] Using the first layout found since no error'
                         ' information is available')
                break

            # Readout errors of the link qubits
            readout_fidelity = np.array(
                [1. - min(readout_errors[layout[qobj.logical_qubit]], 0.99999)
                 for qobj in self.qubit_graph.nodes() if isinstance(qobj, Link)]
            )
            log_error_score = np.sum(np.log(readout_fidelity))
            # 2Q gate errors
            for (gate, logical_qubits), counts in self.magnetic_2q_gate_counts(basis_2q).items():
                qubits = tuple(layout[qubit] for qubit in logical_qubits)
                try:
                    gate_error = gate_errors[(gate, qubits)]
                except KeyError:
                    gate_error = gate_errors[(gate, qubits[::-1])]
                error = min(gate_error, 0.99999)
                log_error_score += np.log(1. - error) * counts
            # If best score, remember the layout
            if score_max is None or log_error_score > score_max:
                score_max = log_error_score
                best_layout = layout
                LOG.info('[layout_heavy_hex] Best layout error score: %.3f', score_max)

        if best_layout is None:
            raise ValueError('I do not think this would ever happen')

        return best_layout

    @abstractmethod
    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        qobj: Link | Plaquette
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
        hamiltonian = SparsePauliOp(link_terms, [-1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [-plaquette_energy] * len(plaquette_terms))
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
        circuit.rz(-2. * time,
                   self.qubit_graph.filter_nodes(lambda qobj: isinstance(qobj, Link)))
        return circuit

    def electric_clifford(self) -> QuantumCircuit:
        """Construct the electric term circuit at delta_t = pi/4."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        circuit.sdg(self.qubit_graph.filter_nodes(lambda qobj: isinstance(qobj, Link)))
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
    def magnetic_clifford(self) -> QuantumCircuit:
        """Construct the magnetic term circuit at K*delta_t = pi/4."""

    @abstractmethod
    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cx'
    ) -> dict[tuple[str, tuple[int, int]], int]:
        """Return a list of (gate name, qubits, counts)."""


def payload_matches(value: Any):
    def match(payload):
        return payload == value
    return match


def payload_contains(values: Any):
    def contains(payload):
        return all(value in payload for value in values)
    return contains
