from typing import Optional
import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from .pure_z2_lgt import PureZ2LGT
from .utils import as_bitarray, to_pauli_string


class PlaquetteDual:
    """Dual lattice of 2d Z2 LGT."""
    def __init__(self, primal: PureZ2LGT, base_link_state: Optional[np.ndarray] = None):
        self._primal = primal
        if base_link_state is None:
            self._base_link_state = np.zeros(primal.num_links)
        else:
            self._base_link_state = np.array(base_link_state)

    def map_link_state(self, link_state: np.ndarray | str) -> np.ndarray:
        link_state = as_bitarray(link_state)
        if np.max(link_state) > 1 or np.min(link_state) < 0:
            raise ValueError('Non-binary link state')
        if link_state.shape[0] != self._base_link_state.shape[0]:
            raise ValueError('Number of links inconsistent with the primal graph')

        # Excited links are those whose states differ from the base
        excited_links = np.nonzero(link_state != self._base_link_state)[0]

        # Cut the dual graph along excited links
        dual_graph = self._primal.dual_graph.copy()
        for link in excited_links:
            dual_graph.remove_edge_from_index(link)
        # Remove boundary links too
        for link, edge_info in dual_graph.edge_index_map().items():
            if None in [dual_graph[node] for node in edge_info[:2]]:
                dual_graph.remove_edge_from_index(link)
        patches = rx.connected_components(dual_graph)  # pylint: disable=no-member

        # Next: Determine which patches are excited.
        # Construct a new graph whose vertices are the patches
        patch_graph = self._primal.dual_graph.copy()
        edge_index_map = patch_graph.edge_index_map()
        # Relabel boundary links
        for link in excited_links:
            if None in [patch_graph[node] for node in edge_index_map[link][:2]]:
                patch_graph.update_edge_by_index(link, -edge_index_map[link][2] - 1)

        for plaquettes in patches:
            plaquettes = list(plaquettes)
            if len(plaquettes) == 1 and patch_graph[plaquettes[0]] is None:
                continue
            patch_graph.contract_nodes(plaquettes, plaquettes)

        # Apply a two-coloring and determine which of 0 or 1 corresponds to excitation
        coloring = rx.two_color(patch_graph)
        # Find a boundary plaquette
        edge_index_map = patch_graph.edge_index_map()
        p1, p2, weight = next(edge_info for edge_info in edge_index_map.values()
                              if None in [patch_graph[node] for node in edge_info[:2]])
        boundary = next(patch for patch in [p1, p2] if patch_graph[patch] is not None)
        if weight > 0:
            base_color = coloring[boundary]
        else:
            base_color = 1 - coloring[boundary]

        # Finally compose the plaquette state
        state = np.zeros(self._primal.num_plaquettes, dtype=int)
        for patch in patch_graph.node_indices():
            if coloring[patch] != base_color and (plaquettes := patch_graph[patch]) is not None:
                state[plaquettes] = 1

        return state

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Construct the Hamiltonian in the plaquette basis."""
        num_p = self._primal.num_plaquettes
        paulis = []
        for p1, p2, _ in self._primal.dual_graph.edge_index_map().values():
            if self._primal.dual_graph[p1] is None:
                paulis.append(to_pauli_string({p2: 'Z'}, num_p))
            elif self._primal.dual_graph[p2] is None:
                paulis.append(to_pauli_string({p1: 'Z'}, num_p))
            else:
                paulis.append(to_pauli_string({p1: 'Z', p2: 'Z'}, num_p))
        coeffs = [1.] * len(paulis)
        paulis += [to_pauli_string({p: 'X'}, num_p) for p in range(num_p)]
        coeffs += [plaquette_energy] * num_p

        return SparsePauliOp(paulis, coeffs)

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        circuit = QuantumCircuit(self._primal.num_plaquettes)
        for p1, p2, _ in self._primal.dual_graph.edge_index_map().values():
            if self._primal.dual_graph[p1] is None:
                circuit.rz(2. * time, p2)
            elif self._primal.dual_graph[p2] is None:
                circuit.rz(2. * time, p1)
            else:
                circuit.rzz(2. * time, p1, p2)
        return circuit

    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self._primal.num_plaquettes)
        circuit.rx(2. * plaquette_energy * time, range(self._primal.num_plaquettes))
        return circuit
