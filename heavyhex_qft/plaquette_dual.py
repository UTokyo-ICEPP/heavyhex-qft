# pylint: disable=no-member
"""Dual lattice of 2d Z2 LGT."""
from typing import Optional
import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from heavyhex_qft.pure_z2_lgt import PureZ2LGT, DummyPlaquette
from heavyhex_qft.utils import as_bitarray, to_pauli_string


class PlaquetteDual:
    """Dual lattice of 2d Z2 LGT."""
    def __init__(self, primal: PureZ2LGT, base_link_state: Optional[np.ndarray] = None):
        self.primal = primal
        if base_link_state is None:
            self.base_link_state = np.zeros(primal.num_links, dtype=np.uint8)
        else:
            assert len(base_link_state) == primal.num_links
            self.base_link_state = np.array(base_link_state, dtype=np.uint8)
        self._base_syndrome = primal.get_syndrome(self.base_link_state)

    @property
    def graph(self) -> rx.PyGraph:
        return self.primal.dual_graph

    @property
    def num_plaquettes(self) -> int:
        return self.primal.num_plaquettes
    
    @property
    def num_links(self) -> int:
        return self.primal.num_links
    
    @property
    def plaquette_ids(self) -> list[int]:
        return self.primal.plaquette_ids
    
    @property
    def link_ids(self) -> list[int]:
        return self.primal.link_ids

    @property
    def base_syndrome(self) -> np.ndarray:
        return self._base_syndrome

    def map_link_state(self, link_state: np.ndarray | str) -> np.ndarray:
        """Interpret a link state as plaquette excitations with respect to the base link state.

        Users must ensure that the link excitations passed to this function form closed loops (link
        states are in the same charge sector as the base link state).

        `i`th bit of link_state should correspond to `nl - 1 - i`th link in primal.link_ids.
        """
        link_state = as_bitarray(link_state)
        if np.max(link_state) > 1 or np.min(link_state) < 0:
            raise ValueError('Non-binary link state')
        if link_state.shape[0] != self.base_link_state.shape[0]:
            raise ValueError('Number of links inconsistent with the primal graph')

        # Excited links are those whose states differ from the base
        # Need link ids -> reverse the bitstring order
        excited_entries = np.nonzero((link_state != self.base_link_state)[::-1])[0]
        link_ids = self.primal.link_ids
        excited_links = set(link_ids[ent] for ent in excited_entries)

        dual_graph = self.graph.copy()
        # Classify the plaquettes as bulk, grounded boundary, or excited boundary
        bulk, grounded, excited = 0, 1, 2
        for node in dual_graph.filter_nodes(lambda node: isinstance(node, int)):
            plaq_id = dual_graph[node]
            for neighbor, _, link_id in dual_graph.in_edges(node):
                if isinstance(dual_graph[neighbor], DummyPlaquette):
                    # Neighbor is dummy -> node is at boundary
                    if link_id in excited_links:
                        dual_graph[node] = (plaq_id, excited)
                    else:
                        dual_graph[node] = (plaq_id, grounded)
                    break
            else:
                dual_graph[node] = (plaq_id, bulk)

        # Cut the dual graph along excited links
        lid_to_edge = {val[2]: edge for edge, val in dual_graph.edge_index_map().items()}
        for link_id in excited_links:
            dual_graph.remove_edge_from_index(lid_to_edge[link_id])
        # Remove dummy plaquettes
        for node in dual_graph.filter_nodes(lambda node: isinstance(node, DummyPlaquette)):
            dual_graph.remove_node(node)

        # Group the plaquettes into connected components
        patches = rx.connected_components(dual_graph)  # pylint: disable=no-member

        # Construct a new graph whose vertices are the patches
        patch_graph = self.graph.copy()
        # Remove dummy plaquettes
        for node in patch_graph.filter_nodes(lambda node: isinstance(node, DummyPlaquette)):
            patch_graph.remove_node(node)
        # Contract the patches with classification
        for nids in patches:
            nids = list(nids)
            plaq_ids = [dual_graph[nid][0] for nid in nids]
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
            
            patch_graph.contract_nodes(nids, (plaq_ids, label))

        # Apply a two-coloring and determine which of 0 or 1 corresponds to excitation
        coloring = rx.two_color(patch_graph)
        for pid, color in coloring.items():
            if patch_graph[pid][1] == grounded:
                excited_color = 1 - color
                break
            if patch_graph[pid][1] == excited:
                excited_color = color
                break
        else:
            raise RuntimeError('No boundary patch found')

        # Finally compose the plaquette state
        pid_to_idx = np.full(self.primal.plaquettes_capacity, -1)
        pid_to_idx[self.plaquette_ids] = np.arange(self.num_plaquettes)[::-1]

        state = np.zeros(self.num_plaquettes, dtype=np.uint8)
        for pid, color in coloring.items():
            if color == excited_color:
                state[pid_to_idx[patch_graph[pid][0]]] = 1

        return state

    def make_hamiltonian(self, plaquette_energy: float) -> SparsePauliOp:
        """Construct the Hamiltonian in the plaquette basis.

        The dual (Gauss's law-resolved) Hamiltonian encodes the charge sector of the base link state
        in the coefficients of the ZZ terms.
        """
        num_p = self.num_plaquettes
        iqs = np.full(self.primal.plaquettes_capacity, -1)
        iqs[self.plaquette_ids] = np.arange(num_p)
        lid_to_idx = np.full(self.primal.links_capacity, -1)
        lid_to_idx[self.link_ids] = np.arange(self.num_links)[::-1]

        paulis = []
        coeffs = []
        for node1, node2, link_id in self.graph.edge_index_map().values():
            p1, p2 = self.graph[node1], self.graph[node2]
            if isinstance(p1, DummyPlaquette):
                paulis.append(to_pauli_string({iqs[p2]: 'Z'}, num_p))
            elif isinstance(p2, DummyPlaquette):
                paulis.append(to_pauli_string({iqs[p1]: 'Z'}, num_p))
            else:
                paulis.append(to_pauli_string({iqs[p1]: 'Z', iqs[p2]: 'Z'}, num_p))
            # Coeff is -1 / +1 if base link state is 0 / 1
            coeffs.append(-1. + 2. * self.base_link_state[lid_to_idx[link_id]])
        paulis += [to_pauli_string({p: 'X'}, num_p) for p in range(num_p)]
        coeffs += [-plaquette_energy] * num_p

        return SparsePauliOp(paulis, coeffs).simplify()

    def electric_evolution(self, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the electric term."""
        iqs = np.full(self.primal.plaquettes_capacity, -1)
        iqs[self.plaquette_ids] = np.arange(self.num_plaquettes)
        iqs = iqs.tolist()
        lid_to_idx = np.full(self.primal.links_capacity, -1)
        lid_to_idx[self.link_ids] = np.arange(self.num_links)[::-1]
        lid_to_idx = lid_to_idx.tolist()

        circuit = QuantumCircuit(self.num_plaquettes)
        for node1, node2, link_id in self.graph.edge_index_map().values():
            angle = (-1. + 2. * self.base_link_state[lid_to_idx[link_id]]) * 2. * time
            p1, p2 = self.graph[node1], self.graph[node2]
            if isinstance(p1, DummyPlaquette):
                circuit.rz(angle, iqs[p2])
            elif isinstance(p2, DummyPlaquette):
                circuit.rz(angle, iqs[p1])
            else:
                circuit.rzz(angle, iqs[p1], iqs[p2])
        return circuit

    def magnetic_evolution(self, plaquette_energy: float, time: float) -> QuantumCircuit:
        """Construct the Trotter evolution circuit of the magnetic term."""
        circuit = QuantumCircuit(self.num_plaquettes)
        circuit.rx(-2. * plaquette_energy * time, range(self.num_plaquettes))
        return circuit
