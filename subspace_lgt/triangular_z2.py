"""Triangular lattice for Z2 pure-gauge Hamiltonian."""
from numbers import Integral
import numpy as np
from qiskit.quantum_info import SparsePauliOp


class TriangularZ2Lattice:
    """Lattice definition and conversion of operator placement to Pauli strings."""
    def __init__(self, configuration):
        self._configuration = configuration

        plaquette_row_starts = [0]
        link_row_starts = [0]
        site_row_starts = [0]
        for row_conf in configuration:
            plaquette_row_starts.append(plaquette_row_starts[-1] + len(row_conf))
            num_horizontal_links = sum(1 for x in row_conf if x == 'v')
            link_row_starts.append(link_row_starts[-1] + num_horizontal_links)
            link_row_starts.append(link_row_starts[-1] + len(row_conf) + 1)
            site_row_starts.append(site_row_starts[-1] + num_horizontal_links + 1)

        num_horizontal_links = sum(1 for x in configuration[-1] if x == '^')
        link_row_starts.append(link_row_starts[-1] + num_horizontal_links)
        site_row_starts.append(site_row_starts[-1] + num_horizontal_links + 1)
        self._plaquette_row_starts = np.array(plaquette_row_starts, dtype=int)
        self._link_row_starts = np.array(link_row_starts, dtype=int)
        self._site_row_starts = np.array(site_row_starts, dtype=int)

    @property
    def num_plaquettes(self):
        return self._plaquette_row_starts[-1]

    @property
    def num_links(self):
        return self._link_row_starts[-1]

    @property
    def num_sites(self):
        return self._site_row_starts[-1]

    def plaquette_id(self, coordinate):
        row, column = coordinate
        if column >= self._plaquette_row_starts[row + 1]:
            raise ValueError(f'Invalid plaquette column number {column} for row {row}')
        return self._plaquette_row_starts[row] + column

    def plaquette_coordinate(self, plaquette_id):
        row = np.searchsorted(self._plaquette_row_starts, plaquette_id, side='right') - 1
        return (row, plaquette_id - self._plaquette_row_starts[row])

    def plaquette_type(self, id_or_coordinate):
        if isinstance(id_or_coordinate, int):
            row, col = self.plaquette_coordinate(id_or_coordinate)
        else:
            row, col = id_or_coordinate

        return self._configuration[row][col]

    def plaquette_links(self, id_or_coordinate):
        if isinstance(id_or_coordinate, Integral):
            row, col = self.plaquette_coordinate(id_or_coordinate)
        else:
            row, col = id_or_coordinate

        if self.plaquette_type((row, col)) == 'v':
            link_coordinates = [
                (row * 2, col // 2),
                (row * 2 + 1, col),
                (row * 2 + 1, col + 1)
            ]
        else:
            link_coordinates = [
                (row * 2 + 1, col),
                (row * 2 + 1, col + 1),
                (row * 2 + 2, col // 2)
            ]
        return [self.link_id(coord) for coord in link_coordinates]

    def link_id(self, coordinate):
        row, column = coordinate
        if column >= self._link_row_starts[row + 1]:
            raise ValueError(f'Invalid column number {column} for row {row}')
        return self._link_row_starts[row] + column

    def link_coordinate(self, link_id):
        row = np.searchsorted(self._link_row_starts, link_id, side='right') - 1
        return (row, link_id - self._link_row_starts[row])

    def link_coordinates(self):
        return sum(([(row, col) for col in range(row_size)]
                    for row, row_size in enumerate(np.diff(self._link_row_starts))),
                   [])

    def site_id(self, coordinate):
        row, column = coordinate
        if column >= self._site_row_starts[row + 1]:
            raise ValueError(f'Invalid site column number {column} for row {row}')
        return self._site_row_starts[row] + column

    def site_coordinate(self, site_id):
        row = np.searchsorted(self._site_row_starts, site_id, side='right') - 1
        return (row, site_id - self._site_row_starts[row])

    def site_links(self, id_or_coordinate):
        if isinstance(id_or_coordinate, Integral):
            row, col = self.site_coordinate(id_or_coordinate)
        else:
            row, col = id_or_coordinate

        link_coordinates = []
        return [self.link_id(coord) for coord in link_coordinates]

    def to_pauli(self, placements):
        """Form the Pauli string corresponding to the given placement of operators in the
        configuration.

        Args:
            placements: {(row, col): operator} or {link_id: operator}
            configuration: Lattice configuration.

        Returns:
            Pauli string corresponding to the operator placement.
        """
        link_paulis = []
        for id_or_coordinate, op in placements.items():
            if isinstance(id_or_coordinate, Integral):
                row, col = self.link_coordinate(id_or_coordinate)
            else:
                row, col = id_or_coordinate
            if col >= self._link_row_starts[row + 1]:
                raise ValueError(f'Invalid column number {col} for row {row}')
            link_paulis.append((self.link_id((row, col)), op.upper()))

        link_paulis = sorted(link_paulis, key=lambda x: x[0])
        pauli = ''
        for link, op in link_paulis:
            pauli = op + ('I' * (link - len(pauli))) + pauli

        pauli = 'I' * (self.num_links - len(pauli)) + pauli
        return pauli

    def make_hamiltonian(self, plaquette_energy):
        link_terms = [self.to_pauli({coord: 'Z'}) for coord in self.link_coordinates()]
        plaquette_terms = []
        for plid in range(self.num_plaquettes):
            plaquette_terms.append(self.to_pauli({lid: 'X' for lid in self.plaquette_links(plid)}))
        hamiltonian = SparsePauliOp(link_terms, [-1.] * len(link_terms))
        hamiltonian += SparsePauliOp(plaquette_terms, [-plaquette_energy] * len(plaquette_terms))
        return hamiltonian
