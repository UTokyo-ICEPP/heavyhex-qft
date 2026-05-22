# pylint: disable=no-member
"""Z2 lattice gauge theory with static charges."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import combinations, count
from typing import Any, Optional, TYPE_CHECKING
import logging
import json
import numpy as np
from scipy.sparse import csc_matrix
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.models import BackendProperties
from heavyhex_qft.elements import Vertex, Link, Plaquette, DummyPlaquette
from heavyhex_qft.utils import as_bitarray, to_pauli_string, qubit_coordinates
if TYPE_CHECKING:
    from heavyhex_qft.plaquette_dual import PlaquetteDual

LOG = logging.getLogger(__name__)


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
    def __init__(self, graph: rx.PyGraph):
        self.graph = graph
        self._make_dual_graph()
        self._make_qubit_graph()
        self.layout = None
        self._set_caches()

    @property
    def num_plaquettes(self) -> int:
        return sum(1 for p in self.graph.attrs['plaquettes'] if p is not None)

    @property
    def num_links(self) -> int:
        return self.graph.num_edges()

    @property
    def num_vertices(self) -> int:
        return self.graph.num_nodes()

    @property
    def num_qubits(self) -> int:
        return self.qubit_graph.num_nodes()

    @property
    def plaquette_ids(self) -> list[int]:
        return [p.id for p in filter(bool, self.graph.attrs['plaquettes'])]

    @property
    def link_ids(self) -> list[int]:
        return [l.id for l in filter(bool, self.graph.attrs['links'])]

    @property
    def vertex_ids(self) -> list[int]:
        return [v.id for v in filter(bool, self.graph.attrs['vertices'])]

    @property
    def plaquettes_capacity(self) -> int:
        return len(self.graph.attrs['plaquettes'])

    @property
    def links_capacity(self) -> int:
        return len(self.graph.attrs['links'])

    @property
    def vertices_capacity(self) -> int:
        return len(self.graph.attrs['vertices'])

    def plaquette_id_to_idx(self, plaq_id: int | list[int]) -> int | list[int]:
        return self._pid_to_idx[plaq_id]

    def link_id_to_idx(self, link_id: int | list[int]) -> int | list[int]:
        return self._lid_to_idx[link_id]

    def vertex_id_to_idx(self, vertex_id: int | list[int]) -> int | list[int]:
        return self._vid_to_idx[vertex_id]

    @property
    def matching_matrix(self) -> csc_matrix:
        return csc_matrix(self._matching_matrix)

    def plaquette_dual(self, base_link_state: Optional[np.ndarray] = None) -> "PlaquetteDual":
        # pylint: disable-next=import-outside-toplevel
        from heavyhex_qft.plaquette_dual import PlaquetteDual
        return PlaquetteDual(self, base_link_state=base_link_state)

    def to_json(self) -> str:
        """Return a JSON serialization of the primal graph."""
        def graph_attrs(attrs):
            json_data = {}
            for key, value in attrs.items():
                if key in ('vertices', 'links', 'plaquettes'):
                    value = json.dumps([None if e is None else e.to_dict() for e in value])
                else:
                    value = self._graph_attr_to_json(key, value)
                json_data[key] = value

            return json_data

        def node_edge_attrs(payload):
            return {'id': json.dumps(payload)}

        data = {'graph': rx.node_link_json(self.graph, graph_attrs=graph_attrs,
                                           node_attrs=node_edge_attrs, edge_attrs=node_edge_attrs),
                'layout': self.layout}
        data |= self._args_json_data()
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> Any:
        def graph_attrs(json_data):
            constructors = {'vertices': Vertex, 'links': Link, 'plaquettes': Plaquette}
            attrs = {}
            for key, value in json_data.items():
                if (ctor := constructors.get(key)):
                    value = [None if e is None else ctor.from_dict(e) for e in json.loads(value)]
                else:
                    value = cls._json_to_graph_attr(key, value)
                attrs[key] = value

            return attrs

        def node_edge_attrs(json_data):
            return json.loads(json_data['id'])

        data = json.loads(data)
        graph = rx.parse_node_link_json(data.pop('graph'), graph_attrs=graph_attrs,
                                        node_attrs=node_edge_attrs, edge_attrs=node_edge_attrs)
        layout = data.pop('layout')
        obj = cls(graph, **data)
        obj.layout = layout
        return obj

    def draw_graph(
        self,
        vertices: Optional[Sequence[int]] = None,
        links: Optional[Sequence[int]] = None,
        vertex_labels: bool = True,
        link_labels: bool = True,
        plaquette_labels: bool = True,
        ax: Optional[Axes] = None
    ) -> Figure | None:
        selected_vertices = set(vertices or [])
        selected_links = set(links or [])

        vertices = self.graph.attrs['vertices']
        kwargs = {'pos': {node: vertices[self.graph[node]].position
                          for node in self.graph.node_indices()}}
        if vertex_labels:
            kwargs['labels'] = str
        if link_labels:
            kwargs['edge_labels'] = str

        if selected_vertices:
            kwargs['node_color'] = ['#1f78b4'] * self.num_vertices
            if len(selected_vertices) == self.num_vertices:
                # selected_vertices is a binary filter
                selected_vertices = np.nonzero(selected_vertices)[0]
            for vid in selected_vertices:
                node = self.graph.find_node_by_weight(vid)
                kwargs['node_color'][node] = '#b41f1f'

        if selected_links:
            kwargs['edge_color'] = ['k'] * self.num_links
            if len(selected_links) == self.num_links:
                # selected_links is a binary filter
                selected_links = np.nonzero(selected_links)[0]
            lid_edge_map = {val[2]: edge for edge, val in self.graph.edge_index_map().items()}
            for lid in selected_links:
                edge = lid_edge_map[lid]
                kwargs['edge_color'][edge] = 'r'

        fig = rx.visualization.mpl_draw(self.graph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()

        if plaquette_labels:
            # Draw the plaquette ids
            for plaquette in filter(bool, self.graph.attrs['plaquettes']):
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
        pos = {}
        for node in self.dual_graph.node_indices():
            if isinstance((payload := self.dual_graph[node]), int):
                pos[node] = self.graph.attrs['plaquettes'][payload].position
            else:
                pos[node] = payload.position
        kwargs = {'pos': pos, 'node_shape': 'h', 'node_color': '#b41f78', 'style': '--'}
        if plaquette_labels:
            kwargs['labels'] = lambda pid: str(pid) if isinstance(pid, int) else ''
        if link_labels:
            kwargs['edge_labels'] = str

        fig = rx.visualization.mpl_draw(self.dual_graph, ax=ax, with_labels=True, **kwargs)
        # There is a bug in mpl_draw - fig should be non-None if ax is, but variable ax is
        # overwritten in the function
        if fig is None:
            fig = plt.gcf()

        if vertex_labels:
            # Draw the vertex ids
            for vertex in filter(bool, self.graph.attrs['vertices']):
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
        qgraph.add_edges_from(self.qubit_graph.weighted_edge_list())

        kwargs = {'font_size': 6, 'font_color': 'w', 'node_shape': 's', 'node_color': '#034c3c',
                  'node_size': 440, 'width': 5., 'edge_color': 'r',
                  'pos': {lq: qobj.position for lq, qobj in enumerate(self.qubit_graph.nodes())}}

        layout = layout or self.layout
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
        coupling_map: CouplingMap,
        layout: Optional[list[int]] = None,
        ax: Optional[Axes] = None,
        links: Optional[Sequence[int]] = None,
        plaquettes: Optional[Sequence[int]] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        **kwargs
    ) -> Figure:
        layout = layout or self.layout
        if not layout:
            raise ValueError('layout required when default is not set')

        cgraph = rx.PyGraph(multigraph=False)
        cgraph.add_nodes_from(coupling_map.graph.node_indices())
        cgraph.add_edges_from(coupling_map.graph.weighted_edge_list())

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

    # def _draw_qubit_graph_links(self, layout, pos, selected_links, ax):
    #     """Draw links on the qubit graph plot."""

    def plaquette_links(self, plaq_id: int) -> list[int]:
        """Return the list of ids of the links surrounding the plaquette."""
        node = self.dual_graph.find_node_by_weight(plaq_id)
        return [val[2] for val in self.dual_graph.in_edges(node)]

    def link_plaquettes(self, link_id: int) -> list[int]:
        """Return the list (size 1 or 2) of ids of the plaquettes that have the link as an edge."""
        nodes = next(val[:2] for val in self.dual_graph.weighted_edge_list() if val[2] == link_id)
        plaq_ids = []
        for node in nodes:
            if isinstance((pid := self.dual_graph[node]), int):
                plaq_ids.append(pid)
        return plaq_ids

    def vertex_links(self, vertex_id: int) -> list[int]:
        """Return the list of ids of the links incident on the vertex."""
        node = self.graph.find_node_by_weight(vertex_id)
        return [val[2] for val in self.graph.in_edges(node)]

    def link_vertices(self, link_id: int) -> tuple[int, int]:
        """Return the ids of the pair of vertices that the link connects."""
        nodes = next(val[:2] for val in self.graph.weighted_edge_list() if val[2] == link_id)
        return [self.graph[node] for node in nodes]

    def link_qubits(self) -> dict[int, int]:
        """Return ids of logical qubits corresponding to links."""
        return {payload.id: lq for lq, payload in enumerate(self.qubit_graph.nodes())
                if isinstance(payload, Link)}

    def plaquette_qubits(self) -> dict[int, int]:
        """Return ids of logical qubits corresponding to plaquettes."""
        return {payload.id: lq for lq, payload in enumerate(self.qubit_graph.nodes())
                if isinstance(payload, Plaquette)}

    def remove_vertex(self, vertex_id: int):
        vertices = self.graph.attrs['vertices']
        node = self.graph.find_node_by_weight(vertex_id)
        vertex = vertices[vertex_id]
        vertices[vertex_id] = None
        for plaq_id in vertex.plaquettes:
            self.graph.attrs['plaquettes'][plaq_id] = None
        for neighbor in self.graph.neighbors(node):
            vertices[self.graph[neighbor]].plaquettes -= vertex.plaquettes
        for val in self.graph.in_edges(node):
            self.graph.attrs['links'][val[2]] = None
        # Remove the vertex from the primal graph
        self.graph.remove_node(node)
        # Remake the dual and qubit graphs
        self._make_dual_graph()
        self._make_qubit_graph()
        # Validate the graph
        for link_id in self.graph.edges():
            if not self.link_plaquettes(link_id):
                raise ValueError(f'Link {link_id} has been isolated by the removal of vertex'
                                 f' {vertex_id}. Isolated links do not participate in dynamics.')

        self._set_caches()

    def remove_plaquette(self, plaq_id: int):
        plaquette = self.graph.attrs['plaquettes'][plaq_id]
        self.graph.attrs['plaquettes'][plaq_id] = None
        # Remove plaquette references from bounding vertices
        for vertex_id in plaquette.vertices:
            self.graph.attrs['vertices'][vertex_id].plaquettes.remove(plaq_id)
        # If the plaquette is connected to a dummy in the dual, remove the link
        node = self.dual_graph.find_node_by_weight(plaq_id)
        for neighbor in self.dual_graph.neighbors(node):
            if isinstance(self.dual_graph[neighbor], DummyPlaquette):
                link_id = self.dual_graph.get_edge_data(node, neighbor)
                n1, n2 = next(val[:2] for val in self.graph.weighted_edge_list()
                              if val[2] == link_id)
                self.graph.remove_edge(n1, n2)
                self.graph.attrs['links'][link_id] = None
        # Remove isolated vertices
        for vertex_id in plaquette.vertices:
            node = self.graph.find_node_by_weight(vertex_id)
            if len(self.graph.neighbors(node)) == 0:
                self.graph.remove_node(node)
        # Remake the dual and qubit graphs
        self._make_dual_graph()
        self._make_qubit_graph()
        self._set_caches()

    def layout_heavy_hex(
        self,
        coupling_map: Optional[CouplingMap] = None,
        qubit_assignment: Optional[int | dict[str, int]] = None,
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
        cgraph.add_edges_from(coupling_map.graph.weighted_edge_list())

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
            layout = np.empty(self.num_qubits, dtype=int)
            for pq, lq in mapping.items():
                layout[lq] = pq

            if not gate_errors:
                best_layout = layout
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

        self.layout = best_layout.tolist()
        return list(self.layout)

    def get_syndrome(self, link_state: np.ndarray | str) -> np.ndarray:
        """Compute the bit-flip syndrome (parity of sum of link 0/1s at each vertex) from a link
        measurement result.

        `i`th bit of link_state counted from the left should represent the state of `nl - i - 1`th
        link, where the link order is given by self.link_ids. In the returned bit string, `j`th bit
        from the left corresponds to `nv - j - 1`th vertex (in self.vertex_ids order).
        """
        return np.sum(as_bitarray(link_state) * self._matching_matrix, axis=1) & 1

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Return the Z2 LGT Hamiltonian expressed as a SparsePauliOp.

        The lengths of the Pauli strings equal the number of links in the lattice, not the number
        of qubits.
        """
        nq = self.num_links
        link_terms = [to_pauli_string({iq: 'Z'}, nq) for iq in range(nq)]
        plaquette_terms = []
        for plaq_id in self.plaquette_ids:
            lids = self.plaquette_links(plaq_id)
            plaquette_terms.append(to_pauli_string({iq: 'X' for iq in self._lid_to_bit[lids]}, nq))

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
        circuit = QuantumCircuit(self.num_qubits)
        circuit.rz(-2. * time, self.link_qubits().values())
        return circuit

    def electric_clifford(self) -> QuantumCircuit:
        """Construct the electric term circuit at delta_t = pi/4."""
        circuit = QuantumCircuit(self.num_qubits)
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

    def measure(self) -> QuantumCircuit:
        """Construct the measurement circuit with creg ordered by link_ids."""
        circuit = QuantumCircuit(self.num_qubits, self.num_links)
        link_qubits = self.link_qubits()
        for ic, link_id in enumerate(self.link_ids):
            circuit.measure(link_qubits[link_id], ic)
        return circuit

    def _make_dual_graph(self):
        """Construct the dual graph of the lattice from the primal graph."""
        vertices = self.graph.attrs['vertices']
        plaquettes = self.graph.attrs['plaquettes']
        # Initialze the dual graph with plaquette nodes.
        self.dual_graph = rx.PyGraph()
        self.dual_graph.add_nodes_from(self.plaquette_ids)

        for node1, node2, link_id in self.graph.weighted_edge_list():
            vid1 = self.graph[node1]
            vid2 = self.graph[node2]
            # Intersection between the sets of plaquette ids surrounding the two vertices
            link_plaq_ids = list(vertices[vid1].plaquettes & vertices[vid2].plaquettes)
            if len(link_plaq_ids) == 2:
                # This link is in between two plaquettes
                self.dual_graph.add_edge(self.dual_graph.find_node_by_weight(link_plaq_ids[0]),
                                         self.dual_graph.find_node_by_weight(link_plaq_ids[1]),
                                         link_id)
            else:
                # This link is at the boundary of the lattice
                # -> Add a new dummy with an edge connecting it to the boundary plaquette
                plaquette = plaquettes[link_plaq_ids[0]]
                ppos = np.array(plaquette.position)
                vpos1 = np.array(vertices[vid1].position)
                vpos2 = np.array(vertices[vid2].position)
                link_center = (vpos1 + vpos2) * 0.5
                position = tuple(2. * link_center - ppos)
                dummy_id = self.dual_graph.add_node(
                    DummyPlaquette(position=position, vertices={vid1, vid2})
                )
                self.dual_graph.add_edge(self.dual_graph.find_node_by_weight(link_plaq_ids[0]),
                                         dummy_id, link_id)

    def _make_qubit_graph(self):
        links = self.graph.attrs['links']
        self.qubit_graph = rx.PyGraph()
        self.qubit_graph.add_nodes_from([links[lid] for lid in self.graph.edges()])
        self._connect_qubit_graph()

    @abstractmethod
    def _connect_qubit_graph(self):
        """Add plaquette qubits and define the qubit connections."""

    def _set_caches(self):
        # Mappings between element ids to array indices (right to left)
        self._lid_to_idx = np.full(self.links_capacity, -1)
        self._lid_to_idx[self.link_ids] = np.arange(self.num_links)[::-1]
        self._pid_to_idx = np.full(self.plaquettes_capacity, -1)
        self._pid_to_idx[self.plaquette_ids] = np.arange(self.num_plaquettes)[::-1]
        self._vid_to_idx = np.full(self.vertices_capacity, -1)
        self._vid_to_idx[self.vertex_ids] = np.arange(self.num_vertices)[::-1]

        # Mappings between element ids to clbits
        self._lid_to_bit = np.full(self.links_capacity, -1)
        self._lid_to_bit[self.link_ids] = np.arange(self.num_links)
        # pid_to_bit is used in the dual lattice Hamiltonian construction
        self._pid_to_bit = np.full(self.plaquettes_capacity, -1)
        self._pid_to_bit[self.plaquette_ids] = np.arange(self.num_plaquettes)

        # Matching matrix for syndrome calculation
        self._matching_matrix = np.zeros((self.num_vertices, self.num_links), dtype=int)
        for irow, vertex_id in enumerate(self.vertex_ids[::-1]):
            vertex_lids = self.vertex_links(vertex_id)
            self._matching_matrix[irow, self._lid_to_idx[vertex_lids]] = 1


    def _graph_attr_to_json(self, key: str, value: Any) -> str:
        return json.dumps(value)

    @classmethod
    def _json_to_graph_attr(cls, key: str, value: str) -> Any:
        return json.loads(value)

    def _args_json_data(self) -> dict[str, Any]:
        return {}
