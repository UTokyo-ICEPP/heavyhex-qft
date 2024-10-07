"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from itertools import count
from qiskit.quantum_info import SparsePauliOp
import rustworkx as rx


class TriangularZ2Lattice:
    """Triangular lattice for pure-Z2 gauge theory."""
    def __init__(self, configuration):
        # Sanitize the configuration string
        config_rows = configuration.split('\n')
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

        for upper, lower in zip(config_rows[:-1], config_rows[1:]):
            if any(u == '*' and l == '*' for u, l in zip(upper, lower)):
                raise ValueError('Invalid lattice configuration')

        self.graph = rx.PyGraph()
        self.graph.add_nodes_from(range(configuration.count('*')))

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

        self.dual_graph = rx.PyGraph()
        self.dual_graph.add_nodes_from([('edge', idx) for idx in self.graph.edge_indices()])

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

                plaq_node_id = self.dual_graph.add_node(('plaq', next(plaq_id_gen)))
                for n1, n2 in endpoints:
                    self.dual_graph.add_edge(
                        list(self.graph.edge_indices_from_endpoints(n1, n2))[0],
                        plaq_node_id,
                        None
                    )

    @property
    def num_plaquettes(self):
        return len(self.dual_graph.filter_nodes(lambda data: data[0] == 'plaq'))

    @property
    def num_links(self):
        return self.graph.num_edges()

    @property
    def num_sites(self):
        return self.graph.num_nodes()

    def plaquette_links(self, plaq_id):
        plaq_node = list(self.dual_graph.filter_nodes(lambda data: data == ('plaq', plaq_id)))[0]
        return list(self.dual_graph.neighbors(plaq_node))

    def site_links(self, site_id):
        return list(self.graph.incident_edges(site_id))

    def to_pauli(self, link_ops):
        """Form the Pauli string corresponding to the given link operators.

        Args:
            link_ops: {link_id: operator}
            configuration: Lattice configuration.

        Returns:
            Pauli string corresponding to the operator placement.
        """
        link_paulis = []
        for link_id, op in link_ops.items():
            link_paulis.append((link_id, op.upper()))

        link_paulis = sorted(link_paulis, key=lambda x: x[0])
        pauli = ''
        for link, op in link_paulis:
            pauli = op + ('I' * (link - len(pauli))) + pauli

        pauli = 'I' * (self.num_links - len(pauli)) + pauli
        return pauli

    def make_hamiltonian(self, plaquette_energy):
        link_terms = [self.to_pauli({link_id: 'Z'}) for link_id in self.graph.edge_indices()]
        plaquette_terms = []
        for plid in range(self.num_plaquettes):
            plaquette_terms.append(self.to_pauli({lid: 'X' for lid in self.plaquette_links(plid)}))
        hamiltonian = SparsePauliOp(link_terms, [-1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [-plaquette_energy] * len(plaquette_terms))
        return hamiltonian
