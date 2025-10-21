# pylint: disable=unused-argument, no-member
"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from numbers import Number
import re
from itertools import permutations
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from .pure_z2_lgt import PureZ2LGT, Vertex, Link, Plaquette, DummyPlaquette


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
        self.configuration = configuration
        config_rows = sanitize_rows(configuration)
        graph, direct_links = make_primal_graph(config_rows)
        dual_graph = make_dual_graph(graph)
        qubit_graph = make_qubit_graph(dual_graph, direct_links)
        super().__init__(graph, dual_graph, qubit_graph)

    def _draw_qubit_graph_links(self, layout, pos, selected_links, ax):
        # Locate one plaquette qubit and compute the coordinate transformation between the dual
        # graph and the physical qubit graph
        plaq_id = self.qubit_graph.filter_nodes(lambda qobj: isinstance(qobj, Plaquette))[0]
        plaquette = self.qubit_graph[plaq_id]
        ref_coord = np.array(plaquette.position)
        offset = np.array(pos[layout[plaquette.logical_qubit]])

        for link_id, (n1, n2, _) in self.graph.edge_index_map().items():
            x1, y1 = 2. * (np.array(self.graph[n1].position) - ref_coord) + offset
            x2, y2 = 2. * (np.array(self.graph[n2].position) - ref_coord) + offset
            color = '#ff11ff' if link_id in selected_links else '#881188'
            ax.plot([x1, x2], [y1, y2],
                    linewidth=1, linestyle='solid', marker='none', color=color)

    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        qobj: Link | Plaquette
    ) -> bool:
        """Node matcher function for qubit mapping."""
        if isinstance(qobj, Plaquette):
            return len(physical_neighbors) == 3
        return len(physical_neighbors) in (1, 2)

    def _twoq_gate_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the parallel 2-qubit gate ordering."""
        nump = self.num_plaquettes
        controls = np.full((nump, 3), -1, dtype=int)
        targets = np.empty(nump, dtype=int)
        invalid_link_id = self.num_links
        for plaquette in self.dual_graph.nodes():
            if (isinstance(plaquette, DummyPlaquette)
                    or (plaquette.logical_qubit is None and plaquette.direct_link is None)):
                continue

            link_ids = self.plaquette_links(plaquette.plaq_id)
            if (dlink := plaquette.direct_link) is None:
                # Standard plaquette - all link qubits connected to the plaquette qubit
                targets[plaquette.plaq_id] = plaquette.logical_qubit
            else:
                targets[plaquette.plaq_id] = dlink.logical_qubit
                # Remove the direct link from the list of control qubits and add a dummy index
                link_ids.remove(dlink.link_id)
                link_ids.append(invalid_link_id)
                invalid_link_id += 1

            for perm in permutations(link_ids):
                # Find a permutation that does not clash with any of the existing control sequence
                if np.all(np.array(perm)[None, :] != controls):
                    controls[plaquette.plaq_id] = perm
                    break
            else:
                raise RuntimeError('Failed to find a viable link ID permutation')

        # Set the invalid controls to -1
        controls[np.where(controls >= self.num_links)] = -1

        return controls, targets

    def magnetic_evolution(
        self,
        plaquette_energy: float,
        time: float,
        basis_2q: str = 'cx'
    ) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
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

        controls, targets = self._twoq_gate_table()
        masks = controls >= 0

        # Rzzz circuit sandwitched by Hadamards on all links
        circuit.h(range(self.num_links))

        if basis_2q == 'cx':
            for control_qubits, mask in zip(controls.T, masks.T):
                circuit.cx(control_qubits[mask].tolist(), targets[mask].tolist())
            circuit.rz(angle, targets)
            for control_qubits, mask in zip(controls.T[::-1], masks.T[::-1]):
                circuit.cx(control_qubits[mask].tolist(), targets[mask].tolist())

        elif basis_2q == 'cz':
            circuit.h(targets)
            for control_qubits, mask in zip(controls.T, masks.T):
                circuit.cz(control_qubits[mask].tolist(), targets[mask].tolist())
            circuit.rx(angle, targets)
            for control_qubits, mask in zip(controls.T[::-1], masks.T[::-1]):
                circuit.cz(control_qubits[mask].tolist(), targets[mask].tolist())
            circuit.h(targets)
        elif basis_2q == 'rzz':
            # The last positive control will be used for Rzz
            rzz_controls = np.empty_like(targets)
            for icol, (control_qubits, mask) in enumerate(zip(controls.T, masks.T)):
                mask_rzz = np.all(controls[:, icol + 1:] == -1, axis=1)
                mask_cx = mask & ~mask_rzz
                if np.any(mask_cx):
                    circuit.cx(control_qubits[mask_cx].tolist(), targets[mask_cx].tolist())
                rzz_controls[mask & mask_rzz] = control_qubits[mask & mask_rzz]
                masks[:, icol] = mask_cx
            if sign_angle < 0.:
                # Continuous Rzz accepts positive arguments only; sandwitch with Xs
                circuit.x(targets)
            circuit.rzz(abs_angle, rzz_controls, targets)
            if sign_angle < 0.:
                circuit.x(targets)
            for control_qubits, mask in zip(controls.T[::-1], masks.T[::-1]):
                if np.any(mask):
                    circuit.cx(control_qubits[mask].tolist(), targets[mask].tolist())
        else:
            raise ValueError(f'Invalid basis_2q: {basis_2q}')

        circuit.h(range(self.num_links))

        return circuit

    def magnetic_clifford(self) -> QuantumCircuit:
        """Construct the magnetic term circuit at K*delta_t = pi/4."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        controls, targets = self._twoq_gate_table()
        masks = controls >= 0

        # Rzzz(pi/2) circuit sandwitched by Hadamards on all links
        circuit.h(range(self.num_links))
        for control_qubits, mask in zip(controls.T, masks.T):
            circuit.cx(control_qubits[mask].tolist(), targets[mask].tolist())
        circuit.sdg(targets)
        for control_qubits, mask in zip(controls.T[::-1], masks.T[::-1]):
            circuit.cx(control_qubits[mask].tolist(), targets[mask].tolist())
        circuit.h(range(self.num_links))

        return circuit

    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cx'
    ) -> dict[tuple[str, tuple[int, int]], int]:
        """Return a list of (gate name, qubits, counts)."""
        gate_counts = defaultdict(int)
        controls, targets = self._twoq_gate_table()
        masks = controls >= 0

        if basis_2q in ['cx', 'cz']:
            for control_qubits, mask in zip(controls.T, masks.T):
                for qc, qt in zip(control_qubits[mask].tolist(), targets[mask].tolist()):
                    gate_counts[(basis_2q, (qc, qt))] += 2

        elif basis_2q == 'rzz':
            rzz_controls = np.empty_like(targets)
            for icol, (control_qubits, mask) in enumerate(zip(controls.T, masks.T)):
                mask_rzz = np.all(controls[:, icol + 1:] == -1, axis=1)
                mask_cx = mask & ~mask_rzz
                for qc, qt in zip(control_qubits[mask_cx].tolist(), targets[mask_cx].tolist()):
                    gate_counts[('cx', (qc, qt))] += 2
                rzz_controls[mask & mask_rzz] = control_qubits[mask & mask_rzz]

            for qc, qt in zip(rzz_controls.tolist(), targets.tolist()):
                gate_counts[('rzz', (qc, qt))] += 1

        return dict(gate_counts)


def sanitize_rows(configuration: str) -> list[str]:
    """Trim whitespaces and tokenize the lattice configuration string to rows."""
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
    max_row_length = max(len(row) for row in rows)
    rows = [row + ' ' * (max_row_length - len(row)) for row in rows]
    return rows


def make_primal_graph(config_rows: list[str]) -> tuple[rx.PyGraph, list[int]]:
    """Construct the primal graph of the lattice from the row-tokenized configuration string.

    The configuration syntax assumes that all adjacent vertices are connected and that the edges
    will be connected via a plaquette qubit in the qubit graph, unless otherwise specified.
    Special-case characters are:
    - "^" or "v" for vertex: In the top and bottom rows, respectively, horizontally adjacent
      vertices may be disconnected if either one is expressed by these characters.
    - "-", "╵", "╷", or "╎" for link: Signifies that the corresponding plaquette will not have a
      plaquette (ancilla) qubit at the center, and instead the link where these characters appear
      will be directly connected to the other two link qubits.
    """
    # Identify vertices and their positions from the configuration
    nodes = {}
    nrows = len(config_rows)
    vertex_id = 0
    for irow, row in enumerate(config_rows):
        ifirst = next(ip for ip, char in enumerate(row) if char != ' ')
        for icol in range(ifirst, len(row), 2):
            char = row[icol]
            if char == ' ':
                continue
            if char not in '*^v':
                raise ValueError(f'Invalid row {row}')
            nodes[(icol, nrows - 1 - irow)] = (vertex_id, char)
            vertex_id += 1

    # Initialize the graph with vertex nodes
    graph = rx.PyGraph()
    graph.add_nodes_from([Vertex(vid, coord) for vid, coord in enumerate(nodes.keys())])

    # Connect the vertices
    direct_links = []
    for vertex in graph.nodes():
        node_char = nodes[vertex.position][1]
        x, y = vertex.position
        irow = nrows - 1 - y
        icol = x
        # From left to right: Both vertices must be *
        if node_char == '*' and (neighbor := nodes.get((x + 2, y), (None, None)))[1] == '*':
            link_id = graph.add_edge(vertex.vertex_id, neighbor[0], None)
            graph.update_edge_by_index(link_id, Link(link_id))
            if config_rows[irow][icol + 1] == '-':
                direct_links.append(link_id)
        # From top to bottom left
        if (neighbor := nodes.get((x - 1, y - 1))) is not None:
            link_id = graph.add_edge(vertex.vertex_id, neighbor[0], None)
            graph.update_edge_by_index(link_id, Link(link_id))
            if config_rows[irow + 1][icol] in '╵╎':
                direct_links.append(link_id)
        # From top to bottom right
        if (neighbor := nodes.get((x + 1, y - 1))) is not None:
            link_id = graph.add_edge(vertex.vertex_id, neighbor[0], None)
            graph.update_edge_by_index(link_id, Link(link_id))
            if config_rows[irow][icol + 1] in '╷╎':
                direct_links.append(link_id)

    return graph, direct_links


def make_dual_graph(primal_graph: rx.PyGraph) -> rx.PyGraph:
    """Construct the dual graph of the lattice from the primal graph."""
    # Find the plaquettes through graph cycles of length 4
    plaquettes = set()
    for node in primal_graph.node_indices():
        for cycle in rx.all_simple_paths(primal_graph, node, node, min_depth=4, cutoff=4):
            if len(cycle) == 4:
                plaquettes.add(tuple(sorted(cycle[:3])))
    plaquettes = sorted(plaquettes)

    # Initialze the dual graph with plaquette nodes.
    # Also update the vertex objects of the primal graph.
    dual_graph = rx.PyGraph()
    for plaq_id, plaq_vids in enumerate(plaquettes):
        vertices = [primal_graph[vid] for vid in plaq_vids]
        # Position the plaquette node at the center of the rectangle that bounds the triangle
        pos_x = np.mean([vertex.position[0] for vertex in vertices])
        pos_y = np.mean(np.unique([vertex.position[1] for vertex in vertices]))
        dual_graph.add_node(Plaquette(plaq_id, (pos_x, pos_y), set(plaq_vids)))
        for vertex in vertices:
            vertex.plaquettes.add(plaq_id)

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
            elif ppos[0] > vpos1[0] or ppos[0] > vpos2[0]:
                # Left
                position = (min(vpos1[0], vpos2[0]), ppos[1])
            else:
                # Right
                position = (max(vpos1[0], vpos2[0]), ppos[1])
            dummy_id = dual_graph.add_node(
                DummyPlaquette(position=position, vertices=set([vid1, vid2]))
            )
            dual_graph.add_edge(plaq_id, dummy_id, link)

    return dual_graph


def make_qubit_graph(dual_graph: rx.PyGraph, direct_links: list[int]):
    """Construct the qubit graph from the dual graph."""
    direct_links = set(direct_links)

    # Initialize the qubit graph from links
    qubit_graph = rx.PyGraph()
    qubit_ids = qubit_graph.add_nodes_from(dual_graph.edges())
    for link, qubit_id in zip(dual_graph.edges(), qubit_ids):
        link.logical_qubit = qubit_id

    # Add plaquette qubits
    for plaquette in dual_graph.nodes():
        if isinstance(plaquette, DummyPlaquette):
            continue
        if (plaq_direct_link := set(dual_graph.incident_edges(plaquette.plaq_id)) & direct_links):
            # If there is a direct link surrounding this plaquette, that is the connection target
            link_id = plaq_direct_link.pop()
            plaquette.direct_link = dual_graph.get_edge_data_by_index(link_id)
        else:
            # Otherwise add a plaquette node and make it the connection target
            qubit_id = qubit_graph.add_node(plaquette)
            plaquette.logical_qubit = qubit_id

    # Make edges from each link to the corresponding target
    for link in dual_graph.edges():
        if link.link_id in direct_links:
            continue
        for plaq_id in dual_graph.get_edge_endpoints_by_index(link.link_id):
            plaquette = dual_graph[plaq_id]
            if isinstance(plaquette, DummyPlaquette):
                continue
            target = plaquette.logical_qubit
            if target is None:
                target = plaquette.direct_link.link_id
            qubit_graph.add_edge(link.link_id, target, None)

    return qubit_graph
