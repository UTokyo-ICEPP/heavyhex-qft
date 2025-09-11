# pylint: disable=no-member
"""Dual lattice of 2d Z2 LGT."""
from typing import Optional
import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from .pure_z2_lgt import PureZ2LGT, DummyPlaquette
from .utils import as_bitarray, to_pauli_string


class PlaquetteDual:
    """Dual lattice of 2d Z2 LGT."""
    def __init__(self, primal: PureZ2LGT, base_link_state: Optional[np.ndarray] = None):
        self.primal = primal
        if base_link_state is None:
            self._base_link_state = np.zeros(primal.num_links)
        else:
            self._base_link_state = np.array(base_link_state)

    @property
    def graph(self) -> rx.PyGraph:
        return self.primal.dual_graph

    @property
    def num_plaquettes(self) -> int:
        return self.primal.num_plaquettes

    def map_link_state(self, link_state: np.ndarray | str) -> np.ndarray:
        """Interpret a link state as plaquette excitations with respect to the base link state."""
        link_state = as_bitarray(link_state)
        if np.max(link_state) > 1 or np.min(link_state) < 0:
            raise ValueError('Non-binary link state')
        if link_state.shape[0] != self._base_link_state.shape[0]:
            raise ValueError('Number of links inconsistent with the primal graph')

        # Excited links are those whose states differ from the base
        excited_links = set(np.nonzero(link_state != self._base_link_state)[0])

        dual_graph = self.graph.copy()
        edge_index_map = dual_graph.edge_index_map()
        # Classify the plaquettes as bulk, grounded boundary, or excited boundary
        bulk, grounded, excited = 0, 1, 2
        for nid in dual_graph.filter_nodes(lambda node: not isinstance(node, DummyPlaquette)):
            plaquette = dual_graph[nid]
            for eid in dual_graph.incident_edges(nid):
                nbid = next(idx for idx in edge_index_map[eid][:2] if idx != nid)
                if isinstance(dual_graph[nbid], DummyPlaquette):
                    if eid in excited_links:
                        dual_graph[nid] = (plaquette, excited)
                    else:
                        dual_graph[nid] = (plaquette, grounded)
                    break
            else:
                dual_graph[nid] = (plaquette, bulk)

        # Cut the dual graph along excited links
        for link in excited_links:
            dual_graph.remove_edge_from_index(link)
        # Remove dummy plaquettes
        for nid in dual_graph.filter_nodes(lambda node: isinstance(node, DummyPlaquette)):
            dual_graph.remove_node(nid)

        # Group the plaquettes into connected components
        patches = rx.connected_components(dual_graph)  # pylint: disable=no-member

        # Construct a new graph whose vertices are the patches
        patch_graph = self.graph.copy()
        # Remove dummy plaquettes
        for nid in patch_graph.filter_nodes(lambda node: isinstance(node, DummyPlaquette)):
            patch_graph.remove_node(nid)
        # Contract the patches with classification
        for nids in patches:
            nids = list(nids)
            is_grounded = any(dual_graph[nid][1] == grounded for nid in nids)
            is_excited = any(dual_graph[nid][1] == excited for nid in nids)
            if is_grounded:
                if is_excited:
                    raise RuntimeError('Inconsistent boundary patch excitation - Wilson non-loops?')
                label = grounded
            elif is_excited:
                label = excited
            else:
                label = bulk
            patch_graph.contract_nodes(nids, (nids, label))

        # Apply a two-coloring and determine which of 0 or 1 corresponds to excitation
        coloring = rx.two_color(patch_graph)
        for pid, color in coloring.items():
            if patch_graph[pid][1] == grounded:
                base_color = color
                break
            if patch_graph[pid][1] == excited:
                base_color = 1 - color
                break
        else:
            raise RuntimeError('No boundary patch found')

        # Finally compose the plaquette state
        state = np.zeros(self.num_plaquettes, dtype=int)
        for pid, color in coloring.items():
            if color != base_color:
                plaq_ids = [dual_graph[nid][0].plaq_id for nid in patch_graph[pid][0]]
                state[plaq_ids] = 1

        return state

    def link_statevector_indices(self) -> np.ndarray:
        """Return the indices of the link state vector."""
        num_p = self.num_plaquettes
        plaquette_links = np.array([self.graph.incident_edges(ip) for ip in range(num_p)])
        link_bits = np.sum(1 << plaquette_links, axis=1)
        bin_indices = (np.arange(2 ** num_p)[:, None] >> np.arange(num_p)[None, :]) % 2
        link_indices = np.bitwise_xor.reduce(bin_indices * link_bits, axis=1)
        link_indices ^= np.sum(1 << self._base_link_state)
        return link_indices

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Construct the Hamiltonian in the plaquette basis."""
        num_p = self.num_plaquettes
        paulis = []
        for nid1, nid2, _ in self.graph.edge_index_map().values():
            p1, p2 = self.graph[nid1], self.graph[nid2]
            if isinstance(p1, DummyPlaquette):
                paulis.append(to_pauli_string({p2.plaq_id: 'Z'}, num_p))
            elif isinstance(p2, DummyPlaquette):
                paulis.append(to_pauli_string({p1.plaq_id: 'Z'}, num_p))
            else:
                paulis.append(to_pauli_string({p1.plaq_id: 'Z', p2.plaq_id: 'Z'}, num_p))
        coeffs = [-1.] * len(paulis)
        paulis += [to_pauli_string({p: 'X'}, num_p) for p in range(num_p)]
        coeffs += [-plaquette_energy] * num_p

        return SparsePauliOp(paulis, coeffs)

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        circuit = QuantumCircuit(self.num_plaquettes)
        for nid1, nid2, _ in self.graph.edge_index_map().values():
            p1, p2 = self.graph[nid1], self.graph[nid2]
            if isinstance(p1, DummyPlaquette):
                circuit.rz(-2. * time, p2.plaq_id)
            elif isinstance(p2, DummyPlaquette):
                circuit.rz(-2. * time, p1.plaq_id)
            else:
                circuit.rzz(-2. * time, p1.plaq_id, p2.plaq_id)
        return circuit

    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.num_plaquettes)
        circuit.rx(-2. * plaquette_energy * time, range(self.num_plaquettes))
        return circuit
