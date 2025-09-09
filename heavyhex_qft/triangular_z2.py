# pylint: disable=unused-argument, no-member
"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from numbers import Number
import re
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from .pure_z2_lgt import PureZ2LGT, payload_matches, payload_contains


class TriangularZ2Lattice(PureZ2LGT):
    r"""Triangular lattice for pure-Z2 gauge theory.

    The constructor takes a string argument representing the structure of the lattice. The string
    should contain only characters '*', '^', 'v', ' ', and '\n', with the non-whitespace characters
    representing the locations of the vertices. Vertices appearing in a single line are aligned
    horizontally. There must be an odd number of whitespaces between the vertex characters.
    Different characters represent the number of edges emanating from the vertex: '*' is a full
    (hexagonal) vertex, and '^' and 'v' are top- and bottom-row vertices with only two edges.
    The placement of asterisks in two consecutive lines must be staggered.

    Examples:
        - Two plaquettes
            lattice = TriangularZ2Lattice('''
             *
            * *
             *
            ''')

        - 14 plaquettes
            lattice = TriangularZ2Lattice('''
             * * * *
            * * * * *
             * * * *
            ''')
        - Fox
            lattice = TriangularZ2Lattice('''
             *   *
            * * * *
             * * *
              * *
               *
            ''')
    """
    def __init__(self, configuration: str):
        config_rows = sanitize_rows(configuration)
        graph = make_primal_graph(config_rows)
        dual_graph = make_dual_graph(graph)
        qubit_graph = make_qubit_graph(graph, dual_graph)
        super().__init__(graph, dual_graph, qubit_graph)

    def _draw_qubit_graph_links(self, graph, layout, pos, selected_links, ax):
        plaq_logical_qubit = self.qubit_graph.filter_nodes(payload_contains(['plaq']))[0]
        plaq_id = self.qubit_graph[plaq_logical_qubit][1]
        plaq_nodes = self.dual_graph[plaq_id]
        ref_coord = (np.mean([self.graph[node][0] for node in plaq_nodes]),
                     np.mean(np.unique([self.graph[node][1] for node in plaq_nodes])))
        offset = np.array(pos[layout[plaq_logical_qubit]])

        for link_id, (n1, n2, _) in self.graph.edge_index_map().items():
            x1, y1 = 2. * (np.array(self.graph[n1]) - ref_coord) + offset
            x2, y2 = 2. * (np.array(self.graph[n2]) - ref_coord) + offset
            color = '#ff11ff' if link_id in selected_links else '#881188'
            ax.plot([x1, x2], [y1, y2],
                    linewidth=1, linestyle='solid', marker='none', color=color)

    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        node_type: str,
        obj_id: int
    ) -> bool:
        """Node matcher function for qubit mapping."""
        if node_type == 'plaq':
            return len(physical_neighbors) == 3
        else:
            return len(physical_neighbors) in (1, 2)

    def _plaquette_links(self):
        """Return a list of link qubits for each plaquette, ordered counterclockwise."""
        plaquette_links = []
        for plid in range(self.num_plaquettes):
            links = list(sorted(self.plaquette_links(plid)))
            if links[2] - links[1] == 1:
                # Downward pointing plaquette
                plaquette_links.append(links)
            else:
                # Upward pointing plaquette
                plaquette_links.append([links[0], links[2], links[1]])
        return np.array(plaquette_links)

    def magnetic_evolution(
        self,
        plaquette_energy: float,
        time: float,
        basis_2q: str = 'cx'
    ) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        plaquette_links = self._plaquette_links()
        # Plaquette qubit ids
        qpl = np.arange(self.num_links, self.qubit_graph.num_nodes())
        # Rzzz rotation angle
        angle = -2. * plaquette_energy * time
        if isinstance(angle, Number):
            angle = (angle + np.pi) % (2. * np.pi) - np.pi
            abs_angle = abs(angle)
            sign_angle = np.sign(angle)
            if basis_2q == 'rzz' and abs_angle > np.pi / 2.:
                raise ValueError(
                    f'Rzz angle {angle} is too large for the Rzz gate; use basis_2q="cx" or "cz"'
                )
        else:
            abs_angle = angle
            sign_angle = -1.

        # Rzzz circuit sandwitched by Hadamards on all links
        circuit.h(range(self.num_links))
        if basis_2q == 'cx':
            circuit.cx(plaquette_links[:, 0], qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            circuit.cx(plaquette_links[:, 2], qpl)
            circuit.rz(angle, qpl)
            circuit.cx(plaquette_links[:, 2], qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            circuit.cx(plaquette_links[:, 0], qpl)
        elif basis_2q == 'cz':
            circuit.h(qpl)
            circuit.cz(plaquette_links[:, 0], qpl)
            circuit.cz(plaquette_links[:, 1], qpl)
            circuit.cz(plaquette_links[:, 2], qpl)
            circuit.rx(angle, qpl)
            circuit.cz(plaquette_links[:, 2], qpl)
            circuit.cz(plaquette_links[:, 1], qpl)
            circuit.cz(plaquette_links[:, 0], qpl)
            circuit.h(qpl)
        else:
            circuit.cx(plaquette_links[:, 0], qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            if sign_angle < 0.:
                # Continuous Rzz accepts positive arguments only; sandwitch with Xs to reverse sign
                circuit.x(qpl)
            circuit.rzz(abs_angle, plaquette_links[:, 2], qpl)
            if sign_angle < 0.:
                circuit.x(qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            circuit.cx(plaquette_links[:, 0], qpl)
        circuit.h(range(self.num_links))
        return circuit

    def magnetic_clifford(self) -> QuantumCircuit:
        """Construct the magnetic term circuit at K*delta_t = pi/4."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        plaquette_links = self._plaquette_links()
        # Plaquette qubit ids
        qpl = np.arange(self.num_links, self.qubit_graph.num_nodes())
        # Rzzz circuit sandwitched by Hadamards on all links
        circuit.h(range(self.num_links))
        circuit.cx(plaquette_links[:, 0], qpl)
        circuit.cx(plaquette_links[:, 1], qpl)
        circuit.cx(plaquette_links[:, 2], qpl)
        circuit.sdg(qpl)
        circuit.cx(plaquette_links[:, 2], qpl)
        circuit.cx(plaquette_links[:, 1], qpl)
        circuit.cx(plaquette_links[:, 0], qpl)
        circuit.h(range(self.num_links))
        return circuit

    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cx'
    ) -> dict[tuple[str, tuple[int, int]], int]:
        """Return a list of (gate name, qubits, counts)."""
        gate_counts = defaultdict(int)
        plaquette_links = self._plaquette_links()
        # Plaquette qubit ids
        qpl = np.arange(self.num_links, self.qubit_graph.num_nodes())
        if basis_2q in ['cx', 'cz']:
            for side in range(3):
                for ctrl, targ in zip(plaquette_links[:, side], qpl):
                    gate_counts[(basis_2q, (ctrl, targ))] += 2

        elif basis_2q == 'rzz':
            for side in range(2):
                for ctrl, targ in zip(plaquette_links[:, side], qpl):
                    gate_counts[('cx', (ctrl, targ))] += 2
            for ctrl, targ in zip(plaquette_links[:, 2], qpl):
                gate_counts[('rzz', (ctrl, targ))] += 1

        return dict(gate_counts)


def sanitize_rows(configuration: str):
    rows = configuration.split('\n')
    if any(re.search('[^ -╷╵╎*^v]', row) for row in rows):
        raise ValueError('Lattice constructor argument contains invalid character(s)')
    first_row = 0
    while not rows[first_row].strip():
        first_row += 1
    rows = rows[first_row:]
    last_row = len(rows)
    while not rows[last_row - 1].strip():
        last_row -= 1
    rows = rows[:last_row]
    first_column = 0
    while all(not row[first_column] for row in rows):
        first_column += 1
    rows = [row[first_column:] for row in rows]
    return rows


def make_primal_graph(config_rows: list[str]):
    nodes = {}
    nrows = len(config_rows)
    for irow, row in enumerate(config_rows):
        ifirst = next(ip for ip, char in enumerate(row) if char != ' ')
        for icol in range(ifirst, len(row), 2):
            char = row[icol]
            if char == ' ':
                continue
            if char not in '*^v':
                raise ValueError(f'Invalid row {row}')
            nodes[(icol, nrows - 1 - irow)] = char

    graph = rx.PyGraph()
    node_indices = dict(zip(nodes.keys(), graph.add_nodes_from(nodes.keys())))

    for (x, y), node_char in nodes.items():
        nid1 = node_indices[(x, y)]
        irow = nrows - 1 - y
        icol = x
        if node_char == '*' and nodes.get((x + 2, y), '') == '*':
            nid2 = node_indices[(x + 2, y)]
            direct_link = config_rows[irow][icol + 1] == '-'
            graph.add_edge(nid1, nid2, direct_link)
        if (nid2 := node_indices.get((x - 1, y - 1))) is not None:
            direct_link = config_rows[irow + 1][icol] in '╵╎'
            graph.add_edge(nid1, nid2, direct_link)
        if (nid2 := node_indices.get((x + 1, y - 1))) is not None:
            direct_link = config_rows[irow][icol + 1] in '╷╎'
            graph.add_edge(nid1, nid2, direct_link)

    return graph


def make_dual_graph(primal_graph: rx.PyGraph):
    # Find the plaquettes through graph cycles of length 4
    plaquettes = set()
    for node in primal_graph.node_indices():
        for cycle in rx.all_simple_paths(primal_graph, node, node, min_depth=4, cutoff=4):
            if len(cycle) == 4:
                plaquettes.add(tuple(sorted(cycle[:3])))
    plaquettes = sorted(plaquettes)

    # Construct the dual graph
    dual_graph = rx.PyGraph()
    dual_graph.add_nodes_from(plaquettes)

    for pid1 in dual_graph.node_indices():
        plaq_nodes = dual_graph[pid1]
        for n1, n2 in zip(plaq_nodes, plaq_nodes[1:] + plaq_nodes[0:1]):
            link_id = primal_graph.edge_indices_from_endpoints(n1, n2)[0]
            if dual_graph.filter_edges(payload_matches(link_id)):
                # Plaquettes already linked
                continue

            neighbors = dual_graph.filter_nodes(payload_contains([n1, n2]))
            neighbors = list(neighbors)
            neighbors.remove(pid1)
            if neighbors:
                pid2 = neighbors[0]
            else:
                pid2 = dual_graph.add_node(tuple(sorted((n1, n2))) + (None,))

            dual_graph.add_edge(pid1, pid2, link_id)

    return dual_graph


def make_qubit_graph(primal_graph: rx.PyGraph, dual_graph: rx.PyGraph):
    qubit_graph = rx.PyGraph()
    qubit_graph.add_nodes_from([('link', link_id) for link_id in primal_graph.edge_indices()])
    for link_id in primal_graph.edge_indices():
        direct_link = primal_graph.get_edge_data_by_index(link_id)
        # Find two plaquettes this link borders
        dual_eid = dual_graph.filter_edges(payload_matches(link_id))[0]
        for plaq_id in dual_graph.get_edge_endpoints_by_index(dual_eid):
            if None in dual_graph[plaq_id]:
                # This is a dummy plaquette node at the boundary
                continue
            # Find the other two links that form this plaquette
            plaq_link_ids = [dual_graph.get_edge_data_by_index(eid)
                             for eid in dual_graph.in_edge_indices(plaq_id)]
            plaq_link_ids.remove(link_id)
            if direct_link:
                for plid in plaq_link_ids:
                    qubit_graph.add_edge(link_id, plid, None)
            elif any(primal_graph.get_edge_data_by_index(plid) for plid in plaq_link_ids):
                # This is a linearly connected (no-ancilla) plaquette
                continue
            else:
                try:
                    plaq_node_id = qubit_graph.filter_nodes(payload_matches(('plaq', plaq_id)))[0]
                except IndexError:
                    plaq_node_id = qubit_graph.add_node(('plaq', plaq_id))
                qubit_graph.add_edge(link_id, plaq_node_id, None)

    return qubit_graph
