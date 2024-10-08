"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from itertools import count
from typing import Union
import numpy as np
from matplotlib.figure import Figure
import rustworkx as rx
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp


class TriangularZ2Lattice:
    r"""Triangular lattice for pure-Z2 gauge theory.

    Hamiltonian of the theory is

    .. math::

        H = -\sum_{e \in \mathcal{E}} Z_e - K \sum_{p \in \mathcal{P}} \prod_{e \in \partial p} X_e.

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

        # Construct the lattice graph (nodes=vertices, edges=links)
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

        # Construct the qubit mapping graph (nodes=links and plaquettes, edges=qubit connectivity)
        self.qubit_graph = rx.PyGraph()
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
                    self.qubit_graph.add_edge(
                        list(self.graph.edge_indices_from_endpoints(n1, n2))[0],
                        plaq_node_id,
                        None
                    )

    @property
    def num_plaquettes(self) -> int:
        return len(self.qubit_graph.filter_nodes(lambda data: data[0] == 'plaq'))

    @property
    def num_links(self) -> int:
        return self.graph.num_edges()

    @property
    def num_vertices(self) -> int:
        return self.graph.num_nodes()

    def draw_graph(self) -> Figure:
        return rx.visualization.mpl_draw(self.graph, with_labels=True, labels=str, edge_labels=str)

    def draw_qubit_graph(self) -> Figure:
        return rx.visualization.mpl_draw(self.qubit_graph, with_labels=True, labels=str)

    def plaquette_links(self, plaq_id: int) -> list[int]:
        """Return the list of node indices in the qubit graph corresponding to the links surrounding
        the plaquette."""
        plaq_node = list(self.qubit_graph.filter_nodes(lambda data: data == ('plaq', plaq_id)))[0]
        return list(self.qubit_graph.neighbors(plaq_node))

    def vertex_links(self, vertex_id: int) -> list[int]:
        """Return the list of node indices in the qubit graph corresponding to the links incident
        on the vertex.

        Note that the edge ids of the lattice graph and the node ids of the corresponding link
        qubits in the coincident.
        """
        return list(self.graph.incident_edges(vertex_id))

    def layout_heavy_hex(
        self,
        coupling_map: CouplingMap,
        qubit_assignment: Union[int, dict[tuple[str, int], int]]
    ) -> list[int]:
        """Return the physical qubit layout of the qubit graph using qubits in the coupling map.

        Args:
            coupling_map: backend.coupling_map.
            qubit_assignment: Physical qubit id to assign link 0 to, or an assignment hint dict of
                form {('link' or 'plaq', id): (physical qubit)}.

        Returns:
            List of physical qubit ids to be passed to the transpiler.
        """
        cgraph = coupling_map.graph.to_undirected()
        for idx in cgraph.node_indices():
            if len(cgraph.neighbors(idx)) == 3:
                cgraph[idx] = (idx, 'plaq')
            else:
                cgraph[idx] = (idx, 'link')

        if isinstance(qubit_assignment, int):
            qubit_assignment = {('link', 0): qubit_assignment}

        def node_matcher(physical_qubit, lattice_qubit):
            # True if this is an assigned qubit
            if qubit_assignment.get(lattice_qubit) == physical_qubit[0]:
                return True
            # Otherwise check the qubit type (plaq or link)
            return physical_qubit[1] == lattice_qubit[0]

        vf2 = rx.vf2_mapping(cgraph, self.qubit_graph, node_matcher=node_matcher, subgraph=True,
                             induced=False)
        try:
            mapping = next(vf2)
        except StopIteration as exc:
            raise ValueError('Layout with the given qubit assignment could not be found.') from exc

        layout = [None] * self.qubit_graph.num_nodes()
        for physical_qubit, logical_qubit in mapping.items():
            layout[logical_qubit] = physical_qubit

        return layout

    def to_pauli(self, link_ops: dict[int, str], pad_plaquettes: bool = False) -> str:
        """Form the Pauli string corresponding to the given link operators.

        If pad_plaquettes is True, ('I' * num_plaquettes) is appended to the returned string.
        """
        link_paulis = []
        for link_id, op in link_ops.items():
            link_paulis.append((link_id, op.upper()))

        link_paulis = sorted(link_paulis, key=lambda x: x[0])
        pauli = ''
        for link, op in link_paulis:
            pauli = op + ('I' * (link - len(pauli))) + pauli

        pauli = 'I' * (self.num_links - len(pauli)) + pauli
        if pad_plaquettes:
            pauli = 'I' * self.num_plaquettes + pauli
        return pauli

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Return the Z2 LGT Hamiltonian expressed as a SparsePauliOp.

        The lengths of the Pauli strings equal the number of links in the lattice, not the number
        of qubits.
        """
        link_terms = [self.to_pauli({link_id: 'Z'}) for link_id in self.graph.edge_indices()]
        plaquette_terms = []
        for plid in range(self.num_plaquettes):
            plaquette_terms.append(self.to_pauli({lid: 'X' for lid in self.plaquette_links(plid)}))
        hamiltonian = SparsePauliOp(link_terms, [-1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [-plaquette_energy] * len(plaquette_terms))
        return hamiltonian

    def charge_subspace(self, vertex_charge: list[int]) -> np.ndarray:
        """Return the dimensions of the full Hilbert space (d=2**num_link) that span the subspace
        of the given vertex charges.

        TODO This implementation is very likely not efficient and unnecessarily restricts the
        usability of the method to smaller number of links.
        """
        if len(vertex_charge) != self.num_vertices or any(c not in (0, 1) for c in vertex_charge):
            raise ValueError(f'Argument must be a length-{self.num_vertices} list of 0 or 1')

        hspace_dim = 2 ** self.num_links
        basis_indices = np.arange(hspace_dim)
        all_bitstrings = np.array([(basis_indices >> i) % 2
                                   for i in range(self.num_links)], dtype='uint8').T
        flt = np.ones(hspace_dim, dtype='uint8')
        for vertex, parity in enumerate(vertex_charge):
            links = self.vertex_links(vertex)
            flt *= np.asarray(np.sum(all_bitstrings[:, links], axis=1) % 2 == parity, dtype='uint8')

        return np.nonzero(flt)[0]

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        circuit = QuantumCircuit(self.qubit_graph.num_nodes())
        circuit.rz(-2. * time, range(self.num_links))
        return circuit

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
