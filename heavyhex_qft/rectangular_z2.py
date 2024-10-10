"""Rectangular lattice for Z2 pure-gauge Hamiltonian."""
from collections import defaultdict
from itertools import count
from typing import Union
import numpy as np
import rustworkx as rx
from qiskit.circuit import QuantumCircuit

from .pure_z2_lgt import PureZ2LGT


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
    def __init__(self, configuration: Union[tuple[int, int], str]):
        if isinstance(configuration, tuple):
            num_vertices = np.prod(configuration)
            node_ids = np.arange(num_vertices).reshape(configuration).tolist()
        else:
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

            num_vertices = configuration.count('*')
            node_id_gen = iter(range(num_vertices))
            node_ids = []
            for row in config_rows:
                node_ids.append([next(node_id_gen) if char == '*' else None for char in row])

        super().__init__(num_vertices)

        # Construct the lattice graph (nodes=vertices, edges=links)
        edge_id_gen = iter(count())
        for upper, lower in zip(node_ids[:-1], node_ids[1:]):
            for left, right in zip(upper[:-1], upper[1:]):
                if left is not None and right is not None:
                    self.graph.add_edge(left, right, next(edge_id_gen))
            for top, bottom in zip(upper, lower):
                if top is not None and bottom is not None:
                    self.graph.add_edge(top, bottom, next(edge_id_gen))

        for left, right in zip(node_ids[-1][:-1], node_ids[-1][1:]):
            if left is not None and right is not None:
                self.graph.add_edge(left, right, next(edge_id_gen))

        # Construct the dual and qubit mapping graphs
        self.qubit_graph.add_nodes_from([('link', idx) for idx in self.graph.edge_indices()])

        # Find the plaquettes through graph cycles of length 5
        # Plaquettes are normalized to be clockwise starting from the earliest node
        plaquettes = set()
        for node in self.graph.node_indices():
            for cycle in rx.all_simple_paths(self.graph, node, node):
                if len(cycle) == 5:
                    cycle = np.roll(cycle[:4], -np.argmin(cycle[:4]))
                    if cycle[-1] < cycle[1]:
                        cycle = np.flip(np.roll(cycle, 3))
                    plaquettes.add(tuple(cycle))

        self.dual_graph.add_nodes_from(range(len(plaquettes)))
        self.qubit_graph.add_nodes_from([('plaq', pid) for pid in range(len(plaquettes))])

        link_plaq_assoc = defaultdict(list)
        anc_id_gen = iter(count())
        for pid, plaquette in enumerate(sorted(plaquettes)):
            # pylint: disable-next=cell-var-from-loop
            plaq_node_id = self.qubit_graph.filter_nodes(lambda d: d == ('plaq', pid))[0]
            link_ids = []
            for inode in range(4):
                lid = self.graph.edge_indices_from_endpoints(
                    plaquette[inode], plaquette[(inode + 1) % 4]
                )[0]
                link_ids.append(lid)
                link_plaq_assoc[lid].append(pid)

            for l1, l2 in [link_ids[:2], link_ids[2:]]:
                anc_node_id = self.qubit_graph.add_node(('anc', next(anc_id_gen)))
                for lid in [l1, l2]:
                    # pylint: disable-next=cell-var-from-loop
                    link_node_id = self.qubit_graph.filter_nodes(lambda d: d == ('link', lid))[0]
                    self.qubit_graph.add_edge(link_node_id, anc_node_id, None)
                self.qubit_graph.add_edge(plaq_node_id, anc_node_id, None)

        for lid in self.graph.edge_indices():
            pids = link_plaq_assoc[lid]
            if len(pids) == 1:
                pids.append(self.dual_graph.add_node(None))
            self.dual_graph.add_edge(pids[0], pids[1], lid)

    def _layout_node_matcher(
        self,
        physical_qubit: int,
        physical_neighbors: tuple[int, ...],
        node_type: str,
        obj_id: int
    ) -> bool:
        """Node matcher function for qubit mapping."""
        if node_type == 'anc':
            return len(physical_neighbors) == 3
        if node_type == 'link':
            # Relying on IBM qubit numbering scheme..
            return (abs(physical_qubit - physical_neighbors[0]) == 1
                    and abs(physical_qubit - physical_neighbors[1]) == 1)
        if node_type == 'plaq':
            return (abs(physical_qubit - physical_neighbors[0]) != 1
                    and abs(physical_qubit - physical_neighbors[1]) != 1)

    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        plaquette_links = np.array([list(sorted(self.plaquette_links(plid)))
                                    for plid in range(self.num_plaquettes)])
        plaquette_node_ids = list(sorted(self.qubit_graph.filter_nodes(lambda d: d[0] == 'plaq')))
        plaquette_ancillas = np.array([sorted(self.qubit_graph.neighbors(nid))
                                      for nid in plaquette_node_ids])
        circuit.h(range(self.num_links))
        for ilink in range(4):
            circuit.cx(plaquette_links[:, ilink], plaquette_ancillas[:, ilink % 2])
        for ianc in range(2):
            circuit.cx(plaquette_ancillas[:, ianc], plaquette_node_ids)
        circuit.rz(-2. * plaquette_energy * time, plaquette_node_ids)
        for ianc in range(1, -1, -1):
            circuit.cx(plaquette_ancillas[:, ianc], plaquette_node_ids)
        for ilink in range(3, -1, -1):
            circuit.cx(plaquette_links[:, ilink], plaquette_ancillas[:, ilink % 2])
        circuit.h(range(self.num_links))
        return circuit
