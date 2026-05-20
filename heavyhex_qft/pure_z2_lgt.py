# pylint: disable=no-member
"""Z2 lattice gauge theory with static charges."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations, count
from typing import Any, Optional, TYPE_CHECKING
import logging
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.models import BackendProperties
from heavyhex_qft.utils import as_bitarray, to_pauli_string, qubit_coordinates
if TYPE_CHECKING:
    from heavyhex_qft.plaquette_dual import PlaquetteDual

LOG = logging.getLogger(__name__)


@dataclass
class Vertex:
    """Vertex data."""
    id: int
    position: tuple[float, float]
    plaquettes: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f'V:{self.id}'

    def to_json_data(self):
        return {
            'id': json.dumps(self.id),
            'position': json.dumps(list(self.position)),
            'plaquettes': json.dumps(list(self.plaquettes))
        }

    @classmethod
    def from_json_data(cls, data):
        return Vertex(
            id=json.loads(data['id']),
            position=tuple(json.loads(data['position'])),
            plaquettes=set(json.loads(data['plaquettes']))
        )


@dataclass
class Link:
    """Link data."""
    id: int
    position: tuple[float, float]

    @property
    def label(self) -> str:
        return f'L:{self.id}'

    def to_json_data(self):
        return {
            'id': json.dumps(self.id),
            'position': json.dumps(list(self.position))
        }

    @classmethod
    def from_json_data(cls, data):
        return Link(
            id=json.loads(data['id']),
            position=tuple(json.loads(data['position']))
        )


@dataclass
class Plaquette:
    """Plaquette data."""
    id: int
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f'P:{self.id}'

    def to_json_data(self):
        return {
            'id': json.dumps(self.id),
            'position': json.dumps(list(self.position)),
            'vertices': json.dumps(list(self.vertices))
        }

    @classmethod
    def from_json_data(cls, data):
        return Plaquette(
            id=json.loads(data['id']),
            position=tuple(json.loads(data['position'])),
            vertices=set(json.loads(data['vertices']))
        )


@dataclass
class DummyPlaquette:
    """Dummy plaquette for dual graph."""
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f''

    def to_json_data(self):
        return {
            'position': json.dumps(list(self.position)),
            'vertices': json.dumps(list(self.vertices))
        }

    @classmethod
    def from_json_data(cls, data):
        return DummyPlaquette(
            position=tuple(json.loads(data['position'])),
            vertices=set(json.loads(data['vertices']))
        )


@dataclass
class Ancilla:
    """Ancilla qubit."""

    def to_json_data(self):
        return {}

    @classmethod
    def from_json_data(cls, data):
        return Ancilla()


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
    def __init__(self, graph: rx.PyGraph, dual_graph: rx.PyGraph):
        self.graph = graph
        self.dual_graph = dual_graph
        self._make_qubit_graph()

        # Cached vertex parity
        self._vertex_parity = None

    def _make_qubit_graph(self):
        self.qubit_graph = rx.PyGraph()
        self.qubit_graph.add_nodes_from(self.graph.edges())
        self._connect_qubit_graph()

    @abstractmethod
    def _connect_qubit_graph(self):
        """Add plaquette qubits and define the qubit connections."""

    @property
    def num_plaquettes(self) -> int:
        return len(self.graph.attrs['plaquettes'])

    @property
    def num_links(self) -> int:
        return self.graph.num_edges()

    @property
    def num_vertices(self) -> int:
        return self.graph.num_nodes()

    @property
    def num_qubits(self) -> int:
        return self.qubit_graph.num_nodes()

    def plaquette_dual(self, base_link_state: Optional[np.ndarray] = None) -> "PlaquetteDual":
        # pylint: disable-next=import-outside-toplevel
        from heavyhex_qft.plaquette_dual import PlaquetteDual
        return PlaquetteDual(self, base_link_state=base_link_state)

    def draw_graph(
        self,
        vertices: Optional[Sequence[int]] = None,
        links: Optional[Sequence[int]] = None,
        vertex_labels: bool = True,
        link_labels: bool = True,
        plaquette_labels: bool = True,
        ax: Optional[Axes] = None
    ) -> Figure | None:
        kwargs = {'pos': {vertex.id: vertex.position for vertex in self.graph.nodes()}}
        if vertex_labels:
            kwargs['labels'] = lambda v: str(v.id)
        if link_labels:
            kwargs['edge_labels'] = lambda l: str(l.id)

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

        fig = rx.visualization.mpl_draw(self.graph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()

        if plaquette_labels:
            # Draw the plaquette ids
            for plaquette in self.graph.attrs['plaquettes'].values():
                x, y = plaquette.position
                fig.axes[0].text(x, y, f'{plaquette.id}', ha='center', va='center')

        if not plt.isinteractive() or ax is None:
            return fig

    def draw_dual_graph(
        self,
        vertex_labels: bool = True,
        link_labels: bool = True,
        plaquette_labels: bool = True,
        ax: Optional[Axes] = None
    ) -> Figure | None:
        kwargs = {'pos': {node: self.dual_graph[node].position
                          for node in self.dual_graph.node_indices()},
                  'node_shape': 'h', 'node_color': '#b41f78', 'style': '--'}
        if plaquette_labels:
            kwargs['labels'] = lambda p: str(p.id) if isinstance(p, Plaquette) else ''
        if link_labels:
            kwargs['edge_labels'] = lambda l: str(l.id)

        fig = rx.visualization.mpl_draw(self.dual_graph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()

        if vertex_labels:
            # Draw the vertex ids
            for vertex in self.graph.nodes():
                x, y = vertex.position
                fig.axes[0].text(x, y, f'{vertex.id}', ha='center', va='center')

        if not plt.isinteractive() or ax is None:
            return fig
        
    def draw_qubit_graph(
        self,
        layout: Optional[list[int]] = None,
        ax: Optional[Axes] = None
    ) -> Figure | None:
        qgraph = rx.PyGraph()
        qgraph.add_nodes_from(enumerate(self.qubit_graph.nodes()))
        qgraph.add_edges_from(self.qubit_graph.edge_index_map().values())

        kwargs = {'font_size': 6, 'font_color': 'w',
                  'pos': {lq: qobj.position
                          for lq, qobj in enumerate(self.qubit_graph.nodes())},
                  'node_shape': 's', 'node_color': '#034c3c', 'node_size': 440, 'style': ':'}
        if layout:
            kwargs['labels'] = lambda p: f'{p[0]}\nq{layout[p[0]]}\n{p[1].label}'
        else:
            kwargs['labels'] = lambda p: f'{p[0]}\n{p[1].label}'

        fig = rx.visualization.mpl_draw(qgraph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()
        if not plt.isinteractive() or ax is None:
            return fig

    def draw_physical_qubits(
        self,
        layout: list[int],
        coupling_map: CouplingMap,
        ax: Optional[Axes] = None,
        links: Optional[Sequence[int]] = None,
        plaquettes: Optional[Sequence[int]] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        **kwargs
    ) -> Figure:
        cgraph = rx.PyGraph(multigraph=False)
        cgraph.add_nodes_from(coupling_map.graph.node_indices())
        cgraph.add_edges_from(coupling_map.graph.edge_index_map().values())

        selected_links = set(links or [])
        selected_plaquettes = set(plaquettes or [])
        selected_qubits = set(physical_qubits or [])

        node_color = [None] * cgraph.num_nodes()
        for qobj, physical_qubit in zip(self.qubit_graph.nodes(), layout):
            cgraph[physical_qubit] = f'{physical_qubit}\n{qobj.label}'
            if isinstance(qobj, Link):
                if qobj.id in selected_links:
                    node_color[physical_qubit] = '#ffaaff'
                else:
                    node_color[physical_qubit] = '#cc11cc'
            else:
                if qobj.id in selected_plaquettes:
                    node_color[physical_qubit] = '#aaffff'
                else:
                    node_color[physical_qubit] = '#11cccc'

        for physical_qubit in set(coupling_map.physical_qubits) - set(layout):
            if physical_qubit in selected_qubits:
                node_color[physical_qubit] = '#cccccc'
            else:
                node_color[physical_qubit] = '#888888'

        pos = {iq: (col, row) for iq, (row, col) in enumerate(qubit_coordinates(coupling_map))}

        kwargs['pos'] = pos
        kwargs.setdefault('node_shape', 's')
        kwargs.setdefault('node_size', 240)
        kwargs.setdefault('node_color', node_color)
        kwargs.setdefault('style', ':')
        kwargs.setdefault('font_size', 8)

        fig = rx.visualization.mpl_draw(cgraph, ax=ax, with_labels=True, labels=str, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()
        # Add link drawings
        # self._draw_qubit_graph_links(layout, pos, selected_links, ax or fig.axes[0])

        if not plt.isinteractive() or ax is None:
            return fig

    def _draw_qubit_graph_links(self, layout, pos, selected_links, ax):
        """Draw links on the qubit graph plot."""

    def plaquette_links(self, plaq_id: int) -> list[int]:
        """Return the list of ids of the links surrounding the plaquette."""
        plaquette = self.graph.attrs['plaquettes'][plaq_id]
        node = self.dual_graph.find_node_by_weight(plaquette)
        edge_index_map = self.dual_graph.incident_edge_index_map(node)
        return list(val[2].id for val in edge_index_map.values())

    def link_plaquettes(self, link_id: int) -> list[int]:
        """Return the list (size 1 or 2) of ids of the plaquettes that have the link as an edge."""
        vid1, vid2 = self.link_vertices(link_id)
        return list(self.graph[vid1].plaquettes & self.graph[vid2].plaquettes)

    def vertex_links(self, vertex_id: int) -> list[int]:
        """Return the list of ids of the links incident on the vertex."""
        return list(self.graph.incident_edges(vertex_id))

    def link_vertices(self, link_id: int) -> tuple[int, int]:
        """Return the ids of the pair of vertices that the link connects."""
        return self.graph.edge_index_map()[link_id][:2]

    def link_qubits(self) -> dict[int, int]:
        """Return ids of logical qubits corresponding to links."""
        return {data.id: lq for lq, data in enumerate(self.qubit_graph.nodes())
                if isinstance(data, Link)}

    def plaquette_qubits(self) -> dict[int, int]:
        """Return ids of logical qubits corresponding to plaquettes."""
        return {data.id: lq for lq, data in enumerate(self.qubit_graph.nodes())
                if isinstance(data, Plaquette)}

    def remove_vertex(self, vertex_id: int):
        for plaq_id in self.graph[vertex_id].plaquettes:
            # Remove the plaquette from attrs
            plaquette = self.graph.attrs['plaquettes'].pop(plaq_id)
            # Replace it with a dummy in dual_graph
            node = self.dual_graph.find_node_by_weight(plaquette)
            edge_index_map = self.dual_graph.incident_edge_index_map(node)
            self.dual_graph.remove_node(node)
            dummy = DummyPlaquette(plaquette.position, vertices=plaquette.vertices)
            node = self.dual_graph.add_node(dummy)
            for _, target, link in edge_index_map.values():
                if isinstance(self.dual_graph[target], Plaquette):
                    self.dual_graph.add_edge(target, node, link)

        # Remove the vertex from the primal graph
        self.graph.remove_node(vertex_id)
        # Validate the graph
        for lid in self.graph.edge_indices():
            if not self.link_plaquettes(lid):
                raise ValueError(f'Link {lid} has been isolated by the removal of vertex'
                                 f' {vertex_id}. Isolated links do not participate in dynamics.')

        # Remake the qubit graph
        self._make_qubit_graph()

    def layout_heavy_hex(
        self,
        coupling_map: Optional[CouplingMap] = None,
        qubit_assignment: Optional[int | dict[tuple[str, int], int]] = None,
        backend_properties: Optional[BackendProperties] = None,
        target: Optional[Target] = None,
        basis_2q: str = 'cz/rzz'
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
            readout_errors = np.minimum([target['measure'][(iq,)].error
                                         for iq in range(target.num_qubits)],
                                        0.99999)
            for inst, qargs in target.instructions:
                if inst.num_qubits != 2:
                    continue
                if inst.name not in gate_errors:
                    gate_errors[inst.name] = np.full((target.num_qubits, target.num_qubits),
                                                     0.99999)
                error = min(target[inst.name][qargs].error, 0.99999)
                gate_errors[inst.name][qargs] = error
                gate_errors[inst.name][qargs[::-1]] = error

        elif backend_properties:
            readout_errors = np.array([backend_properties.readout_error(iq)
                                       for iq in range(len(backend_properties.qubits))])
            for gate_prop in backend_properties.gates:
                if len(gate_prop.qubits) != 2:
                    continue
                if gate_prop.gate not in gate_errors:
                    gate_errors[gate_prop.gate] = np.full((target.num_qubits, target.num_qubits),
                                                          0.99999)
                try:
                    error = next(param.value for param in gate_prop.parameters
                                 if param.name == 'gate_error')
                except StopIteration:
                    error = 0.99999
                error = min(error, 0.99999)
                gate_errors[gate_prop.gate][tuple(gate_prop.qubits)] = error
                gate_errors[gate_prop.gate][tuple(gate_prop.qubits[::-1])] = error

        cgraph = rx.PyGraph(multigraph=False)
        cgraph.add_nodes_from(coupling_map.graph.node_indices())
        cgraph.add_edges_from(coupling_map.graph.edge_index_map().values())

        if qubit_assignment is None:
            LOG.info('[layout_heavy_hex] qubit_assignment not given. Using all candidates.')
            node_matcher = None
        else:
            LOG.info('[layout_heavy_hex] Qubit assignment: %s', qubit_assignment)
            def node_matcher(physical_qubit, qobj):
                return ((assignment := qubit_assignment.get(qobj.label)) is None
                        or assignment == physical_qubit)

        LOG.info('[layout_heavy_hex] Finding layout candidates with vf2_mapping..')
        mappings = list(rx.vf2_mapping(cgraph, self.qubit_graph, node_matcher=node_matcher,
                                       subgraph=True, induced=False))
        LOG.info('[layout_heavy_hex] Done.')
        if len(mappings) == 0:
            raise ValueError('Layout with the given qubit assignment could not be found.')

        gate_qubits = {}
        gate_counts = {}
        for gate, qubits_counts in self.magnetic_2q_gate_counts(basis_2q).items():
            qubits = ([], [])
            counts = []
            for logical_qubits, nop in qubits_counts.items():
                qubits[0].append(logical_qubits[0])
                qubits[1].append(logical_qubits[1])
                counts.append(nop)
            
            gate_qubits[gate] = qubits
            gate_counts[gate] = np.array(counts)
        
        link_logical_qubits = [lq for lq, qobj in enumerate(self.qubit_graph.nodes())
                               if isinstance(qobj, Link)]

        score_max, best_layout = None, None
        for mapping in mappings:
            layout = np.empty(self.qubit_graph.num_nodes(), dtype=int)
            for pq, lq in mapping.items():
                layout[lq] = pq 

            if not gate_errors:
                best_layout = layout.tolist()
                LOG.info('[layout_heavy_hex] Using the first layout found since no error'
                         ' information is available')
                break

            # Readout errors of the link qubits
            error_score = np.sum(np.log(1. - readout_errors[layout[link_logical_qubits]]))
            # 2Q gate errors
            for gate, qlists in gate_qubits.items():
                physical_qubits = (layout[qlists[0]], layout[qlists[1]])
                error_score += np.sum(np.log(1. - gate_errors[gate][physical_qubits])
                                      * gate_counts[gate])
            
            # If best score, remember the layout
            if score_max is None or error_score > score_max:
                score_max = error_score
                best_layout = layout
                LOG.info('[layout_heavy_hex] Best layout error score: %.3f', score_max)

        if best_layout is None:
            raise ValueError('I do not think this would ever happen')

        return best_layout.tolist()

    def get_syndrome(self, link_state: np.ndarray | str) -> np.ndarray:
        """Compute the bit-flip syndrome (parity of sum of link 0/1s at each vertex) from a link
        measurement result.
        """
        rev_link_state = np.zeros(self.graph.attrs['max_link_id'] + 1, dtype=np.uint8)
        rev_link_state[self.graph.edge_indices()] = as_bitarray(link_state)[::-1]
        # Return in reverse order (vertex 0 at last bit)
        return np.array([np.sum(rev_link_state[self.vertex_links(vid)]) % 2
                         for vid in self.graph.node_indices()[::-1]], dtype=np.uint8)

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Return the Z2 LGT Hamiltonian expressed as a SparsePauliOp.

        The lengths of the Pauli strings equal the number of links in the lattice, not the number
        of qubits.
        """
        nq = self.num_links
        link_id_to_iq = {link_id: iq for iq, link_id in enumerate(self.graph.edge_indices())}
        link_terms = [to_pauli_string({iq: 'Z'}, nq) for iq in link_id_to_iq.values()]
        plaquette_terms = []
        for plaq_id in self.graph.attrs['plaquettes']:
            iqs = [link_id_to_iq[lid] for lid in self.plaquette_links(plaq_id)]
            plaquette_terms.append(to_pauli_string({iq: 'X' for iq in iqs}, nq))

        hamiltonian = SparsePauliOp(link_terms, [-1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [-plaquette_energy] * len(plaquette_terms))
        return hamiltonian

    def charge_subspace(self, vertex_charge: np.ndarray) -> np.ndarray:
        """Return the indices of the link states that span the subspace of the given vertex charges.
        The vertex charges are given in the reverse order (vertex 0 at the last bit).

        There can be top-down and bottom-up approaches for constructing the subspace. The top-down
        approach would be to generate the full list of link configurations and then filter out the
        states satisfying the vertex charge constraints. This will always require allocating an
        array of 2**nl x nl integers, which is inpractical for nl >~ 35.

        The bottom-up approach will instead start with a null bitstring and build the list of link
        states vertex-by-vertex. The implementation is a lot more complicated in this case, but this
        will be the most memory-efficient construction.
        """
        raise NotImplementedError('charge_subspace is not maintained any more')
    
        if self.num_links > 63:
            raise NotImplementedError('charge_subspace method is only available for lattices with'
                                      ' <= 63 links')
        vertex_charge = np.array(vertex_charge, dtype=np.uint8)
        if vertex_charge.shape[0] != self.num_vertices or np.any(vertex_charge > 1):
            raise ValueError(f'Argument must be a length-{self.num_vertices} list of 0 or 1')

        # Bit patterns of link excitations - final shape will be (subspace_dim, num_links)
        # Axis 1 is ordered from link 0 to nl-1
        mask_all = np.full((1, self.num_links), False)
        covered_lids = set()
        for ivert, parity in enumerate(vertex_charge[::-1]):
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
        circuit.rz(-2. * time, self.link_qubits().values())
        return circuit

    def electric_clifford(self) -> QuantumCircuit:
        """Construct the electric term circuit at delta_t = pi/4."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        circuit.sdg(self.link_qubits().values())
        return circuit

    @abstractmethod
    def magnetic_evolution(
        self,
        plaquette_energy: float,
        time: float,
        basis_2q: str = 'cz/rzz'
    ) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""

    @abstractmethod
    def magnetic_clifford(self) -> QuantumCircuit:
        """Construct the magnetic term circuit at K*delta_t = pi/4."""

    @abstractmethod
    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cz/rzz'
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
