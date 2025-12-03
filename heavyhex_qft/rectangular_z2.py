# pylint: disable=unused-argument, no-member
"""Rectangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from itertools import count
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from heavyhex_qft.pure_z2_lgt import PureZ2LGT, Vertex, Link, Plaquette, DummyPlaquette, Ancilla


class RectangularZ2Lattice(PureZ2LGT):
    r"""Rectangular lattice for pure-Z2 gauge theory.

    The lattice can be constructed from a 2-tuple (M, N) or a string representing the structure of
    the lattice. In the former case, a rectangular lattice with M rows and N columns will be
    created. If initialized with a string, it should contain only characters '*', ' ', and '\n',
    with the asterisks representing the locations of the vertices. Vertices are placed in a
    rectangular grid. Adjacent vertices (both horizontally and vertically) are connected.

    Examples:
        - 2x2 square
            lattice = RectangularZ2Lattice('''
            **
            **
            ''')

        - Swiss
            lattice = RectangularZ2Lattice('''
             **
            ****
            ****
             **
            ''')
    """
    def __init__(self, configuration: tuple[int, int] | str):
        if isinstance(configuration, tuple):
            config_rows = ['*' * configuration[1]] * configuration[0]
        else:
            config_rows = _sanitize_rows(configuration)

        graph, dual_graph = _make_primal_and_dual_graphs(config_rows)
        qubit_graph = _make_qubit_graph(dual_graph)
        super().__init__(graph, dual_graph, qubit_graph)

    def _draw_qubit_graph_links(self, layout, pos, selected_links, ax):
        pass

    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        qobj: Link | Plaquette | Ancilla
    ) -> bool:
        """Node matcher function for qubit mapping."""
        if isinstance(qobj, Ancilla):
            return len(physical_neighbors) == 3
        return len(physical_neighbors) == 2

    def magnetic_evolution(
        self,
        plaquette_energy: float,
        time: float,
        basis_2q: str = 'cx'
    ) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        def rotation(circuit, plaquette_qubits):
            circuit.rz(-2. * plaquette_energy * time, plaquette_qubits)

        return self._magnetic_evolution(rotation)

    def magnetic_clifford(self) -> QuantumCircuit:
        """Construct the magnetic term circuit at K*delta_t = pi/4."""
        def rotation(circuit, plaquette_qubits):
            circuit.sdg(plaquette_qubits)

        return self._magnetic_evolution(rotation)

    def _magnetic_evolution(self, rotation):
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        plaquette_qubits = np.array([plaq.logical_qubit for plaq in self.dual_graph.nodes()
                                     if isinstance(plaq, Plaquette)])
        plaquette_links = np.array([list(sorted(self.plaquette_links(plid)))
                                    for plid in range(self.num_plaquettes)])
        plaquette_ancillas = np.array([sorted(self.qubit_graph.neighbors(nid))
                                      for nid in plaquette_qubits])

        # Rzzz(pi/2) circuit sandwitched by Hadamards on all links
        circuit.h(range(self.num_links))
        for ilink in range(4):
            circuit.cx(plaquette_links[:, ilink], plaquette_ancillas[:, ilink // 2])
        for ianc in range(2):
            circuit.cx(plaquette_ancillas[:, ianc], plaquette_qubits)
        rotation(circuit, plaquette_qubits)
        for ianc in range(1, -1, -1):
            circuit.cx(plaquette_ancillas[:, ianc], plaquette_qubits)
        for ilink in range(3, -1, -1):
            circuit.cx(plaquette_links[:, ilink], plaquette_ancillas[:, ilink // 2])
        circuit.h(range(self.num_links))
        return circuit

    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cx'
    ) -> dict[tuple[str, tuple[int, int]], int]:
        """Return a list of (gate name, qubits, counts)."""
        gate_counts = defaultdict(int)
        plaquette_qubits = np.array([plaq.logical_qubit for plaq in self.dual_graph.nodes()])
        plaquette_links = np.array([list(sorted(self.plaquette_links(plid)))
                                    for plid in range(self.num_plaquettes)])
        plaquette_ancillas = np.array([sorted(self.qubit_graph.neighbors(nid))
                                      for nid in plaquette_qubits])

        for links, ancillas in zip(plaquette_links, plaquette_ancillas):
            for ilink in range(4):
                gate_counts[('cx', (links[ilink], ancillas[ilink // 2]))] += 2
        for plaq, ancillas in zip(plaquette_qubits, plaquette_ancillas):
            for anc in ancillas:
                gate_counts[('cx', (anc, plaq))] += 2

        return dict(gate_counts)


def _sanitize_rows(configuration: str) -> list[str]:
    """Trim whitespaces and tokenize the lattice configuration string to rows."""
    config_rows = configuration.split('\n')
    if any(row.replace('*', '').strip() for row in config_rows):
        raise ValueError('Lattice constructor argument contains invalid character(s)')
    first_row = 0
    while not config_rows[first_row].strip():
        first_row += 1
    config_rows = config_rows[first_row:]
    last_row = len(config_rows)
    while not config_rows[last_row - 1].strip():
        last_row -= 1
    config_rows = config_rows[:last_row]
    first_column = 0
    while all(not row[first_column] for row in config_rows):
        first_column += 1
    config_rows = [row[first_column:] for row in config_rows]
    last_column = max(len(row) for row in config_rows)
    while all(len(row) < last_column or not row[last_column - 1] for row in config_rows):
        last_column -= 1
    config_rows = [row[:last_column] for row in config_rows]
    config_rows = [row + (' ' * (last_column - len(row))) for row in config_rows]
    return config_rows


def _make_primal_and_dual_graphs(config_rows: list[str]) -> tuple[rx.PyGraph, rx.PyGraph]:
    """Construct the primal graph of the lattice from the row-tokenized configuration string."""
    # Identify vertices and their positions from the configuration
    primal_graph = rx.PyGraph()
    dual_graph = rx.PyGraph()
    vid_gen = iter(count())
    pid_gen = iter(count())
    nrows = len(config_rows)
    coord_vids = {}
    for irow, row in enumerate(config_rows):
        for icol, char in enumerate(row):
            if char == ' ':
                continue
            coord = (icol, nrows - 1 - irow)
            vertex = Vertex(next(vid_gen), coord)
            coord_vids[coord] = vertex.vertex_id
            primal_graph.add_node(vertex)
            if (vid_up := coord_vids.get((coord[0], coord[1] + 1))) is not None:
                link_id = primal_graph.add_edge(vid_up, vertex.vertex_id, None)
                primal_graph.update_edge_by_index(link_id, Link(link_id))
            if (vid_left := coord_vids.get((coord[0] - 1, coord[1]))) is not None:
                link_id = primal_graph.add_edge(vid_left, vertex.vertex_id, None)
                primal_graph.update_edge_by_index(link_id, Link(link_id))
            if ((vid_upleft := coord_vids.get((coord[0] - 1, coord[1] + 1))) is not None
                    and vid_up is not None and vid_left is not None):
                pcoord = (coord[0] - 0.5, coord[1] + 0.5)
                vertices = {vid_upleft, vid_up, vid_left, vertex.vertex_id}
                plaquette = Plaquette(next(pid_gen), pcoord, vertices=vertices)
                dual_graph.add_node(plaquette)
                for vid in vertices:
                    primal_graph[vid].plaquettes.add(plaquette.plaq_id)

    # Iterate through the links in the primal graph in the original order so that edge indices
    # in the dual graph coincides with the link ids
    edge_index_map = primal_graph.edge_index_map()
    for link_id in primal_graph.edge_indices():
        vid1, vid2, link = edge_index_map[link_id]
        # Intersection between the sets of plaquette ids surrounding the two vertices
        plaq_ids = primal_graph[vid1].plaquettes & primal_graph[vid2].plaquettes
        if len(plaq_ids) == 2:
            # This link is in between two plaquettes
            dual_graph.add_edge(plaq_ids.pop(), plaq_ids.pop(), link)
        else:
            # This link is at the boundary of the lattice
            # -> Add a new dummy plaquette and an edge that connects it to the boundary plaquette
            plaq_id = plaq_ids.pop()
            ppos = dual_graph[plaq_id].position
            vpos1 = primal_graph[vid1].position
            vpos2 = primal_graph[vid2].position
            if ((ppos[1] < vpos1[1] and ppos[1] < vpos2[1])
                    or (ppos[1] > vpos1[1] and ppos[1] > vpos2[1])):
                # Dummy plaquette at top or bottom
                position = (ppos[0], 2 * vpos1[1] - ppos[1])
            else:
                # Left or right
                position = (2 * vpos1[0] - ppos[0], ppos[1])
            dummy_id = dual_graph.add_node(
                DummyPlaquette(position=position, vertices=set([vid1, vid2]))
            )
            dual_graph.add_edge(plaq_id, dummy_id, link)

    return primal_graph, dual_graph


def _make_qubit_graph(dual_graph: rx.PyGraph) -> rx.PyGraph:
    """Construct the qubit graph from the dual graph."""
    qubit_graph = rx.PyGraph()
    qubit_ids = qubit_graph.add_nodes_from(dual_graph.edges())
    for link, qubit_id in zip(dual_graph.edges(), qubit_ids):
        link.logical_qubit = qubit_id

    # Add plaquette qubits
    for plaq_id in dual_graph.filter_nodes(lambda node: isinstance(node, Plaquette)):
        plaquette = dual_graph[plaq_id]
        qubit_id = qubit_graph.add_node(plaquette)
        plaquette.logical_qubit = qubit_id

    # Add edges and ancilla qubits
    for plaq_id in dual_graph.filter_nodes(lambda node: isinstance(node, Plaquette)):
        plaquette = dual_graph[plaq_id]
        link_ids = sorted(dual_graph.incident_edges(plaq_id))
        for link_pair in [link_ids[:2], link_ids[2:]]:
            qubit_id = qubit_graph.add_node(None)
            qubit_graph[qubit_id] = Ancilla(qubit_id)
            qubit_graph.add_edge(qubit_id, plaquette.logical_qubit, None)
            qubit_graph.add_edge(qubit_id, link_pair[0], None)
            qubit_graph.add_edge(qubit_id, link_pair[1], None)

    return qubit_graph
