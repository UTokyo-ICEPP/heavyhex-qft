# pylint: disable=unused-argument
"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from itertools import count
from numbers import Number
import re
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from .pure_z2_lgt import PureZ2LGT


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
        # Sanitize the configuration string
        config_rows = configuration.split('\n')
        if any(re.search('[^ -*^v>]', row) for row in config_rows):
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
        self.configuration = '\n'.join(config_rows)

        if any(re.search('[*^v>][*^v>]', row) for row in config_rows):
            raise ValueError('Adjacent vertices')
        for upper, lower in zip(config_rows[:-1], config_rows[1:]):
            if any(u not in ' -' and l not in ' -' for u, l in zip(upper, lower)):
                raise ValueError('Lattice rows not staggered')

        super().__init__(len(re.findall('[*^v>]', configuration)))

        # Construct the lattice graph (nodes=vertices, edges=links)
        node_id_gen = iter(self.graph.node_indices())
        edge_id_gen = iter(count())
        self._row_nodes = []
        for row in config_rows:
            row_nodes = []
            for icol, char in enumerate(row):
                if char in '*^v>':
                    row_nodes.append((icol, next(node_id_gen)))
                if char == '*' and icol > 1 and row[icol - 2] == '*':
                    self.graph.add_edge(row_nodes[-2][1], row_nodes[-1][1], next(edge_id_gen))
            self._row_nodes.append(row_nodes)

        for iupper, ilower in zip(range(len(config_rows) - 1), range(1, len(config_rows))):
            # Overlay the two rows - staggering is guaranteed above
            overlaid = [None] * last_column
            for icol, node_id in self._row_nodes[iupper]:
                overlaid[icol] = node_id
            for icol, node_id in self._row_nodes[ilower]:
                overlaid[icol] = node_id
            for inode1, inode2 in zip(overlaid[:-1], overlaid[1:]):
                if inode1 is not None and inode2 is not None:
                    self.graph.add_edge(inode1, inode2, next(edge_id_gen))

        # TODO SAVE PLAQUETTE TYPE FROM - and >

        # Construct the qubit mapping graph (nodes=links and plaquettes, edges=qubit connectivity)
        self.qubit_graph.add_nodes_from([('link', idx) for idx in self.graph.edge_indices()])

        # Find the plaquettes through graph cycles of length 4
        plaquettes = set()
        for node in self.graph.node_indices():
            for cycle in rx.all_simple_paths(self.graph, node, node, min_depth=4, cutoff=4):
                if len(cycle) == 4:
                    plaquettes.add(tuple(sorted(cycle[:3])))

        for pid, plaquette in enumerate(sorted(plaquettes)):
            plaq_node_id = self.qubit_graph.add_node(('plaq', pid))
            for inode in range(3):
                link_id = self.graph.edge_indices_from_endpoints(
                    plaquette[inode], plaquette[(inode + 1) % 3]
                )[0]
                self.qubit_graph.add_edge(link_id, plaq_node_id, None)

                # TODO add edge differently for no-ancilla plaquettes

        # Construct the dual graph
        self.dual_graph.add_nodes_from(range(len(plaquettes)))
        for link_id in self.graph.edge_indices():
            # pylint: disable-next=cell-var-from-loop
            link_node = self.qubit_graph.filter_nodes(lambda d: d == ('link', link_id))[0]
            plaq_nodes = self.qubit_graph.neighbors(link_node)
            if len(plaq_nodes) == 0:
                continue
            pidx1 = self.qubit_graph[plaq_nodes[0]][1]
            if len(plaq_nodes) == 1:
                pidx2 = self.dual_graph.add_node(None)
            else:
                pidx2 = self.qubit_graph[plaq_nodes[1]][1]
            self.dual_graph.add_edge(pidx1, pidx2, link_id)

    def _graph_node_pos(self) -> dict[int, tuple[int, int]]:
        pos = {}
        nrows = len(self._row_nodes)
        for irow, row_nodes in enumerate(self._row_nodes):
            for icol, node_id in row_nodes:
                pos[node_id] = (0.5 * icol, np.sqrt(3) * (nrows - irow - 1))
        return pos

    def _draw_qubit_graph_links(self, graph, layout, pos, selected_links, ax):
        layout_r = {qubit: link for link, qubit in enumerate(layout[:self.num_links])}
        plotted_qubits = set()
        for pidx in range(self.num_plaquettes):
            qp = layout[self.num_links + pidx]
            neighbors = graph.neighbors(qp)
            qh = next(ql for ql in neighbors if abs(ql - qp) != 1)
            if qh > qp:
                left_slope = 1
                right_slope = -1
            else:
                left_slope = -1
                right_slope = 1

            for iq, dx, slope in [(qh, 2, 0), (qp - 1, 1, left_slope), (qp + 1, 1, right_slope)]:
                if iq in neighbors and iq not in plotted_qubits:
                    plotted_qubits.add(iq)
                    link = layout_r[iq]
                    color = '#ff11ff' if link in selected_links else '#881188'
                    x, y = pos[iq]
                    ax.plot([x - dx, x + dx], [y - slope * 1, y + slope * 1],
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
