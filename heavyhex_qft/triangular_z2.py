# pylint: disable=unused-argument
"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from itertools import count
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from .pure_z2_lgt import PureZ2LGT


class TriangularZ2Lattice(PureZ2LGT):
    r"""Triangular lattice for pure-Z2 gauge theory.

    The constructor takes a string argument representing the structure of the lattice. The string
    should contain only characters '*', ' ', and '\n', with the asterisks representing the locations
    of the vertices. Vertices appearing in a single line are aligned horizontally. There must be an
    odd number of whitespaces between the asterisks, with a single space indicating the existence
    of a horizontal link between the vertices. The placement of asterisks in two consecutive lines
    must be staggered.

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

        if any('**' in row for row in config_rows):
            raise ValueError('Adjacent vertices')
        for upper, lower in zip(config_rows[:-1], config_rows[1:]):
            if any(u == '*' and l == '*' for u, l in zip(upper, lower)):
                raise ValueError('Lattice rows not staggered')

        super().__init__(configuration.count('*'))

        # Construct the lattice graph (nodes=vertices, edges=links)
        node_id_gen = iter(self.graph.node_indices())
        node_ids = []
        for row in config_rows:
            node_ids.append([next(node_id_gen) if char == '*' else None for char in row])

        edge_id_gen = iter(count())
        for upper, lower in zip(node_ids[:-1], node_ids[1:]):
            for ipos, left in enumerate(upper[:-2]):
                if left is not None and (right := upper[ipos + 2]) is not None:
                    self.graph.add_edge(left, right, next(edge_id_gen))
            for ipos, top in enumerate(upper):
                if top is None:
                    continue
                if ipos > 0 and (bottom := lower[ipos - 1]) is not None:
                    self.graph.add_edge(top, bottom, next(edge_id_gen))
                if ipos < len(lower) - 1 and (bottom := lower[ipos + 1]) is not None:
                    self.graph.add_edge(top, bottom, next(edge_id_gen))

        for ipos, left in enumerate(node_ids[-1][:-2]):
            if left is not None and (right := node_ids[-1][ipos + 2]) is not None:
                self.graph.add_edge(left, right, next(edge_id_gen))

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

        # Construct the dual graph
        self.dual_graph.add_nodes_from(range(len(plaquettes)))
        for link_id in self.graph.edge_indices():
            # pylint: disable-next=cell-var-from-loop
            link_node = self.qubit_graph.filter_nodes(lambda d: d == ('link', link_id))[0]
            plaq_nodes = self.qubit_graph.neighbors(link_node)
            pidx1 = self.qubit_graph[plaq_nodes[0]][1]
            if len(plaq_nodes) == 1:
                pidx2 = self.dual_graph.add_node(None)
            else:
                pidx2 = self.qubit_graph[plaq_nodes[1]][1]
            self.dual_graph.add_edge(pidx1, pidx2, link_id)

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
        # Rzzz circuit sandwitched by Hadamards on all links
        circuit.h(range(self.num_links))
        if basis_2q == 'cx':
            circuit.cx(plaquette_links[:, 0], qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            circuit.cx(plaquette_links[:, 2], qpl)
            circuit.rz(-2. * plaquette_energy * time, qpl)
            circuit.cx(plaquette_links[:, 2], qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            circuit.cx(plaquette_links[:, 0], qpl)
        elif basis_2q == 'cz':
            circuit.h(qpl)
            circuit.cz(plaquette_links[:, 0], qpl)
            circuit.cz(plaquette_links[:, 1], qpl)
            circuit.cz(plaquette_links[:, 2], qpl)
            circuit.rx(-2. * plaquette_energy * time, qpl)
            circuit.cz(plaquette_links[:, 2], qpl)
            circuit.cz(plaquette_links[:, 1], qpl)
            circuit.cz(plaquette_links[:, 0], qpl)
            circuit.h(qpl)
        elif basis_2q == 'rzz':
            circuit.cx(plaquette_links[:, 0], qpl)
            circuit.cx(plaquette_links[:, 1], qpl)
            if time > 0.:
                # Continuous Rzz accepts positive arguments only; sandwitch with Xs to reverse sign
                circuit.x(qpl)
            circuit.rzz(2. * plaquette_energy * time, plaquette_links[:, 2], qpl)
            if time > 0.:
                circuit.x(qpl)
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
