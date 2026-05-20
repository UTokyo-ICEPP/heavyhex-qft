# pylint: disable=unused-argument, no-member
"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from numbers import Number
import re
from itertools import permutations
from typing import Optional
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from heavyhex_qft.pure_z2_lgt import PureZ2LGT, Vertex, Link, Plaquette, DummyPlaquette


class TriangularZ2Lattice(PureZ2LGT):
    r"""Triangular lattice for pure-Z2 gauge theory.

    The constructor argument is a 2-tuple of integers or a string representing the structure of the
    lattice. The string should contain only characters '*', '^', 'v', ' ', and '\n', with the
    non-whitespace characters representing the locations of the vertices. Vertices appearing in a
    single line are aligned horizontally. There must be an odd number of whitespaces between the
    vertex characters. Different characters represent the number of edges emanating from the vertex:
    '*' is a full (hexagonal) vertex, and '^' and 'v' are top- and bottom-row vertices with only two
    edges. The placement of asterisks in two consecutive lines must be staggered.

    When a 2-tuple is provided, it will be interpreted as a pseudo-rectangular stack of
    (rows, columns) triangles as in the examples below.

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
        - (4, 2)
            * *
             * *
            * *
             * *
            * *
        - (3, 5)
            * * * *
             * * *
            * * * *
             * * *

    """
    def __init__(
        self,
        configuration: tuple[int, int] | str | rx.PyGraph
    ):
        if isinstance(configuration, rx.PyGraph):
            graph = configuration.copy()  # TODO copy() returns a shallow copy. Is this dangerous?
        else:
            if isinstance(configuration, tuple):
                row_even = ' '.join(['*'] * ((configuration[1] - 1) // 2 + 2))
                row_odd = ' ' + ' '.join(['*'] * (configuration[1] // 2 + 1))
                configuration = '\n'.join(row_even if i % 2 == 0 else row_odd
                                        for i in range(configuration[0] + 1))

            graph = make_primal_graph(configuration)

        super().__init__(graph)

    def _connect_qubit_graph(self):
        links = self.graph.attrs['links']
        # Add plaquette qubits and connect the qubit nodes
        for plaquette in self.graph.attrs['plaquettes'].values():
            node = self.dual_graph.find_node_by_weight(plaquette.id)
            link_ids = [val[2] for val in self.dual_graph.incident_edge_index_map(node).values()]
            if (joint_link_id := self.graph.attrs['joint_link'].get(plaquette.id)) is None:
                # This plaquette has an associated qubit
                target_qubit = self.qubit_graph.add_node(plaquette)
            else:
                # This plaquette is implemented with link qubits only
                joint_link = links[joint_link_id]
                target_qubit = self.qubit_graph.find_node_by_weight(joint_link)
                link_ids.remove(joint_link_id)

            for link_id in link_ids:
                link_qubit = self.qubit_graph.find_node_by_weight(links[link_id])
                self.qubit_graph.add_edge(link_qubit, target_qubit, None)

    # def _draw_qubit_graph_links(self, layout, pos, selected_links, ax):
    #     # Locate one plaquette qubit and compute the coordinate transformation between the dual
    #     # graph and the physical qubit graph
    #     logical_qubit = self.qubit_graph.filter_nodes(lambda qobj: isinstance(qobj, Plaquette))[0]
    #     plaquette = self.qubit_graph[logical_qubit]
    #     ref_coord = np.array(plaquette.position)
    #     offset = np.array(pos[layout[logical_qubit]])

    #     for link_id, (n1, n2, _) in self.graph.edge_index_map().items():
    #         x1, y1 = 2. * (np.array(self.graph[n1].position) - ref_coord) + offset
    #         x2, y2 = 2. * (np.array(self.graph[n2].position) - ref_coord) + offset
    #         color = '#ff11ff' if link_id in selected_links else '#881188'
    #         ax.plot([x1, x2], [y1, y2],
    #                 linewidth=1, linestyle='solid', marker='none', color=color)

    def _twoq_gate_table(self) -> dict[int, tuple[int, int, int]]:
        """Return the parallel 2-qubit gate ordering."""
        links = self.graph.attrs['links']
        nump = self.num_plaquettes
        controls = np.full((nump, 3), -1, dtype=int)
        targets = np.empty(nump, dtype=int)
        invalid_qubit = self.qubit_graph.num_nodes()
        logical_qubits = {qobj.label: lq for lq, qobj in enumerate(self.qubit_graph.nodes())}
        for itarg, plaquette in enumerate(self.graph.attrs['plaquettes'].values()):
            link_ids = self.plaquette_links(plaquette.id)
            if (joint_link_id := self.graph.attrs['joint_link'].get(plaquette.id)) is None:
                # Standard plaquette - all link qubits connected to the plaquette qubit
                targets[itarg] = logical_qubits[plaquette.label]
                link_qubits = [logical_qubits[links[link_id].label] for link_id in link_ids]
            else:
                joint_link = links[joint_link_id]
                targets[itarg] = logical_qubits[joint_link.label]
                # Remove the direct link from the list of control qubits and add a dummy index
                link_qubits = [logical_qubits[links[link_id].label] for link_id in link_ids
                               if link_id != joint_link_id]
                link_qubits.append(invalid_qubit)
                invalid_qubit += 1

            for perm in permutations(link_qubits):
                # Find a permutation that does not clash with any of the existing control sequence
                if np.all(np.array(perm)[None, :] != controls):
                    controls[itarg] = perm
                    break
            else:
                raise RuntimeError('Failed to find a viable link ID permutation')

        # Set the invalid controls to -1
        controls[np.where(controls >= self.num_links)] = -1
        return {t: tuple(c) for t, c in zip(targets, controls)}

    def magnetic_evolution(
        self,
        plaquette_energy: float,
        time: float,
        basis_2q: str = 'cz/rzz'
    ) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        # Rzzz rotation angle
        angle = -2. * plaquette_energy * time
        if isinstance(angle, Number):
            angle = (angle + np.pi) % (2. * np.pi) - np.pi
            abs_angle = abs(angle)
            sign_angle = np.sign(angle)
            if basis_2q[3:] == 'rzz' and abs_angle > np.pi / 2.:
                raise ValueError(
                    f'Rzz angle {angle} is too large for the Rzz gate; use basis_2q="cx" or "cz"'
                )
        else:
            abs_angle = angle
            sign_angle = -1.

        gate_table = self._twoq_gate_table()
        link_qubits = list(self.link_qubits().values())

        circuit = QuantumCircuit(self.num_qubits)
        # Rzzz circuit sandwitched by Hadamards on all links
        circuit.h(link_qubits)

        if basis_2q[3:] == 'rzz':
            # Last operation for each plaquette is rzz
            last_controls = {t: next(i for i in reversed(range(3)) if cs[i] != -1)
                             for t, cs in gate_table.items()}
        else:
            last_controls = {t: -1 for t in gate_table}

        if basis_2q[:2] == 'cz':
            # Transform CZ to CX
            circuit.h(gate_table.keys())

        for iop in range(3):
            controls = [cs[iop] for t, cs in gate_table.items()
                        if iop != last_controls[t] and cs[iop] != -1]
            if controls:
                targets = [t for t, cs in gate_table.items() if cs[iop] != -1]
                if basis_2q[:2] == 'cx':
                    circuit.cx(controls, targets)
                else:
                    circuit.cz(controls, targets)

            controls = [cs[iop] for t, cs in gate_table.items() if iop == last_controls[t]]
            if controls:
                targets = [t for t in gate_table if iop == last_controls[t]]
                if basis_2q[:2] == 'cz':
                    circuit.h(targets)
                if sign_angle < 0.:
                    # Continuous Rzz accepts positive arguments only; sandwitch with Xs
                    circuit.x(targets)
                circuit.rzz(abs_angle, controls, targets)
                if sign_angle < 0.:
                    circuit.x(targets)
                if basis_2q[:2] == 'cz':
                    circuit.h(targets)

        if basis_2q == 'cx':
            circuit.rz(angle, gate_table.keys())
        elif basis_2q == 'cz':
            circuit.rx(angle, gate_table.keys())

        for iop in reversed(range(3)):
            controls = [cs[iop] for t, cs in gate_table.items()
                        if iop != last_controls[t] and cs[iop] != -1]
            if controls:
                targets = [t for t, cs in gate_table.items() if cs[iop] != -1]
                if basis_2q[:2] == 'cx':
                    circuit.cx(controls, targets)
                else:
                    circuit.cz(controls, targets)

        if basis_2q[:2] == 'cz':
            # Transform CZ to CX
            circuit.h(gate_table.keys())

        circuit.h(link_qubits)

        return circuit

    def magnetic_clifford(self) -> QuantumCircuit:
        """Construct the magnetic term circuit at K*delta_t = pi/4."""
        gate_table = self._twoq_gate_table()
        link_qubits = list(self.link_qubits().values())

        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        # Rzzz(pi/2) circuit sandwitched by Hadamards on all links
        circuit.h(link_qubits)
        for iop in range(3):
            controls = [cs[iop] for cs in gate_table.values() if cs[iop] != -1]
            targets = [t for t, cs in gate_table.items() if cs[iop] != -1]
            circuit.cx(controls, targets)
        circuit.sdg(gate_table.keys())
        for iop in reversed(range(3)):
            controls = [cs[iop] for cs in gate_table.values() if cs[iop] != -1]
            targets = [t for t, cs in gate_table.items() if cs[iop] != -1]
            circuit.cx(controls, targets)
        circuit.h(link_qubits)

        return circuit

    def magnetic_2q_gate_counts(
        self,
        basis_2q: str = 'cz/rzz'
    ) -> dict[str, dict[tuple[int, int], int]]:
        """Return a list of (gate name, qubits, counts)."""
        gate_counts = {basis_2q[:2]: defaultdict(int)}
        if basis_2q[3:] == 'rzz':
            gate_counts['rzz'] = defaultdict(int)
        gate_table = self._twoq_gate_table()

        if basis_2q[3:] == 'rzz':
            # Last operation for each plaquette is rzz
            last_controls = {t: next(i for i in reversed(range(3)) if cs[i] != -1)
                             for t, cs in gate_table.items()}
        else:
            last_controls = {t: -1 for t in gate_table}

        for iop in range(3):
            controls = [cs[iop] for t, cs in gate_table.items()
                        if iop != last_controls[t] and cs[iop] != -1]
            targets = [t for t, cs in gate_table.items() if cs[iop] != -1]
            for c, t in zip(controls, targets):
                gate_counts[basis_2q[:2]][(c, t)] += 2

            controls = [cs[iop] for t, cs in gate_table.items() if iop == last_controls[t]]
            targets = [t for t in gate_table if iop == last_controls[t]]
            for c, t in zip(controls, targets):
                gate_counts[basis_2q[3:]][(c, t)] += 1

        return gate_counts


def make_primal_graph(configuration: str) -> tuple[rx.PyGraph, list[int]]:
    graph = _make_primal_graph(_sanitize_rows(configuration))
    graph.attrs['configuration'] = configuration
    return graph


def _sanitize_rows(configuration: str) -> list[str]:
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


def _make_primal_graph(config_rows: list[str]) -> tuple[rx.PyGraph, list[int]]:
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
    graph = rx.PyGraph(attrs={})
    graph.attrs['vertices'] = {vid: Vertex(vid, coord) for vid, coord in enumerate(nodes.keys())}
    graph.add_nodes_from(graph.attrs['vertices'].keys())
    graph.attrs['max_vertex_id'] = graph.num_nodes() - 1

    def add_link(vertex, neighbor):
        link_id = graph.add_edge(vertex.id, neighbor[0], None)
        graph.update_edge_by_index(link_id, link_id)
        x, y = vertex.position
        nx, ny = graph.attrs['vertices'][neighbor[0]].position
        position = ((x + nx) * 0.5, (y + ny) * 0.5)
        graph.attrs['links'][link_id] = Link(link_id, position)
        return link_id

    # Connect the vertices. For plaquettes with no central qubit, save the id of the link that
    # connects the other two links of the plaquette.
    graph.attrs['links'] = {}
    joint_links = []
    for vertex in graph.attrs['vertices'].values():
        node_char = nodes[vertex.position][1]
        x, y = vertex.position
        irow = nrows - 1 - y
        icol = x
        # From left to right: Both vertices must be *
        if node_char == '*' and (neighbor := nodes.get((x + 2, y), (None, None)))[1] == '*':
            link_id = add_link(vertex, neighbor)
            if config_rows[irow][icol + 1] == '-':
                joint_links.append(link_id)
        # From top to bottom left
        if (neighbor := nodes.get((x - 1, y - 1))) is not None:
            link_id = add_link(vertex, neighbor)
            if config_rows[irow + 1][icol] in '╵╎':
                joint_links.append(link_id)
        # From top to bottom right
        if (neighbor := nodes.get((x + 1, y - 1))) is not None:
            link_id = add_link(vertex, neighbor)
            if config_rows[irow][icol + 1] in '╷╎':
                joint_links.append(link_id)

    graph.attrs['max_link_id'] = graph.num_edges() - 1

    # Find the plaquettes through graph cycles of length 4
    plaq_vlists = set()
    for node in graph.node_indices():
        for cycle in rx.all_simple_paths(graph, node, node, min_depth=4, cutoff=4):
            if len(cycle) == 4:
                plaq_vlists.add(tuple(sorted(cycle[:3])))

    graph.attrs['plaquettes'] = {}
    graph.attrs['max_plaq_id'] = len(plaq_vlists) - 1

    for plaq_id, plaq_vids in enumerate(sorted(plaq_vlists)):
        vertices = [graph.attrs['vertices'][vid] for vid in plaq_vids]
        # Position the plaquette node at the center of the rectangle that bounds the triangle
        pos_x = float(np.mean([vertex.position[0] for vertex in vertices]))
        pos_y = float(np.mean(np.unique([vertex.position[1] for vertex in vertices])))
        plaquette = Plaquette(plaq_id, (pos_x, pos_y), set(plaq_vids))
        graph.attrs['plaquettes'][plaq_id] = plaquette
        for vertex in vertices:
            vertex.plaquettes.add(plaq_id)

    # Add qubit connectivity information
    graph.attrs['joint_link'] = {}
    index_map = graph.edge_index_map()
    for link_id in joint_links:
        v1, v2 = [graph.attrs['vertices'][vid] for vid in index_map[link_id][:2]]
        # There should be at most one plaq_id in the overlap. It's just easier to write as a loop
        for plaq_id in v1.plaquettes & v2.plaquettes:
            graph.attrs['joint_link'][plaq_id] = link_id

    return graph
