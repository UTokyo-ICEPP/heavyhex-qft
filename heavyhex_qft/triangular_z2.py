"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from collections.abc import Callable
from itertools import count
from typing import Any
import numpy as np
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

        plaq_id_gen = iter(count())
        for upper, lower in zip(node_ids[:-1], node_ids[1:]):
            for ipos, top in enumerate(upper[:-1]):
                endpoints = None
                if (top is None and ipos > 0 and (left := upper[ipos - 1]) is not None
                        and (right := upper[ipos + 1]) is not None
                        and (bottom := lower[ipos]) is not None):
                    endpoints = [(left, right), (right, bottom), (bottom, left)]

                if (top is not None and ipos > 0 and (left := lower[ipos - 1]) is not None
                        and (right := lower[ipos + 1]) is not None):
                    endpoints = [(left, right), (right, top), (top, left)]

                if not endpoints:
                    continue

                plaq_node_id = self.qubit_graph.add_node(('plaq', next(plaq_id_gen)))
                for n1, n2 in endpoints:
                    link_id = list(self.graph.edge_indices_from_endpoints(n1, n2))[0]
                    self.qubit_graph.add_edge(link_id, plaq_node_id, None)

        # Construct the dual graph
        self.dual_graph.add_nodes_from(range(next(plaq_id_gen)))
        for link_id in self.graph.edge_indices():
            link_node = list(self.qubit_graph.filter_nodes(lambda d: d == ('link', link_id)))[0]
            plaq_nodes = self.qubit_graph.neighbors(link_node)
            pidx1 = self.qubit_graph[plaq_nodes[0]][1]
            if len(plaq_nodes) == 1:
                pidx2 = self.dual_graph.add_node(None)
            else:
                pidx2 = self.qubit_graph[plaq_nodes[1]][1]
            self.dual_graph.add_edge(pidx1, pidx2, link_id)

    def _layout_node_matcher(
        self,
        qubit_assignment: dict[tuple[str, int], int]
    ) -> Callable[[Any, Any], bool]:

        def node_matcher(physical_qubit_data, lattice_qubit_data):
            physical_qubit, physical_neighbors = physical_qubit_data
            node_type, _ = lattice_qubit_data
            # True if this is an assigned qubit
            if qubit_assignment.get(lattice_qubit_data) == physical_qubit:
                return True
            # Otherwise check the qubit type (plaq or link)
            if node_type == 'plaq':
                num_neighbors = 3
            else:
                num_neighbors = 2
            return len(physical_neighbors) == num_neighbors

        return node_matcher

    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        plaquette_links = np.array([list(sorted(self.plaquette_links(plid)))
                                    for plid in range(self.num_plaquettes)])
        qpl = np.arange(self.num_links, self.qubit_graph.num_nodes())
        circuit.h(range(self.num_links))
        circuit.cx(plaquette_links[:, 0], qpl)
        circuit.cx(plaquette_links[:, 1], qpl)
        circuit.cx(plaquette_links[:, 2], qpl)
        circuit.rz(-2. * plaquette_energy * time, qpl)
        circuit.cx(plaquette_links[:, 2], qpl)
        circuit.cx(plaquette_links[:, 1], qpl)
        circuit.cx(plaquette_links[:, 0], qpl)
        circuit.h(range(self.num_links))
        return circuit
