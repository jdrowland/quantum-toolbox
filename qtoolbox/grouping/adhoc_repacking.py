"""Ad-hoc repacking algorithm."""

from typing import Optional
import heapq

from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.core.group import GroupCollection, PauliGroup
from qtoolbox.core.pauli import PauliString


def adhoc_repacking(
    hamiltonian: Hamiltonian,
    baseline_groups: GroupCollection,
    k: Optional[int] = None
) -> GroupCollection:
    """Ad-hoc repacking: greedily add Paulis to later groups where they commute."""
    # Create new collection starting from baseline
    repacked = GroupCollection()
    for baseline_group in baseline_groups.groups:
        new_group = PauliGroup()
        for pauli in baseline_group.paulis:
            new_group.add(pauli.copy())
        repacked.groups.append(new_group)

    # Initialize group representatives by simulating the insertion order
    # This matches how sorted_insertion builds groups
    n_qubits = hamiltonian.num_qubits()
    all_bits_valid = (1 << n_qubits) - 1

    # Rebuild group representatives as if inserting in order
    for group in repacked.groups:
        if len(group.paulis) == 0:
            group.gx = 0
            group.gz = 0
            group.valid = all_bits_valid
            continue

        # First Pauli initializes the group
        first = group.paulis[0]
        gx = first.x_bits
        gz = first.z_bits
        valid = all_bits_valid

        # Each subsequent Pauli updates the representatives
        for pauli in group.paulis[1:]:
            x, z = pauli.x_bits, pauli.z_bits
            anticommute = (x & gz) ^ (z & gx)

            # Update representatives
            gx |= x
            gz |= z
            valid &= ~anticommute

        group.gx = gx
        group.gz = gz
        group.valid = valid

    # Track which groups each Pauli has been added to
    pauli_to_groups = {}
    for group_idx, group in enumerate(repacked.groups):
        for pauli in group.paulis:
            if pauli not in pauli_to_groups:
                pauli_to_groups[pauli] = []
            pauli_to_groups[pauli].append(group_idx)

    # Priority queue: (-priority, term_idx, pauli)
    # Priority = c²/N_i where N_i is current number of measurements
    pq = []
    for term_idx, term in enumerate(hamiltonian.terms):
        n_measurements = len(pauli_to_groups.get(term, [1]))
        priority = abs(term.coeff) ** 2 / n_measurements
        heapq.heappush(pq, (-priority, term_idx, term))

    # Process queue
    while pq:
        neg_priority, term_idx, pauli = heapq.heappop(pq)

        # Find current maximum group index this Pauli is in
        current_groups = pauli_to_groups.get(pauli, [])
        if not current_groups:
            continue
        max_group_idx = max(current_groups)

        # Try to add to groups after max_group_idx
        added = False
        for group_idx in range(max_group_idx + 1, repacked.num_groups()):
            group = repacked.groups[group_idx]

            x = pauli.x_bits
            z = pauli.z_bits
            support = x | z

            # Fast path: check anticommutation against group representatives
            anticommute = (x & group.gz) ^ (z & group.gx)
            invalid_support = support & ~group.valid

            # If no invalid support and anticommute bits are all in invalid region, can add
            if invalid_support == 0 and (anticommute & group.valid) == 0:
                group.add(pauli.copy())
                # No need to update valid since anticommute was zero in valid region
                # (valid stays the same)
                pauli_to_groups[pauli].append(group_idx)
                added = True
                break

            # Slow path: check against all terms in group
            can_add = True
            if k is None:
                # Full commutation
                can_add = all(pauli.commutes_with(p) for p in group.paulis)
            else:
                # k-commutation
                can_add = all(pauli.k_commutes(p, k) for p in group.paulis)

            if can_add:
                group.add(pauli.copy())
                # Update valid field - shrink it based on anticommutation
                group.valid &= ~anticommute
                pauli_to_groups[pauli].append(group_idx)
                added = True
                break

        # Re-add with updated priority if we added to a new group
        if added:
            n_measurements = len(pauli_to_groups[pauli])
            priority = abs(pauli.coeff) ** 2 / n_measurements
            heapq.heappush(pq, (-priority, term_idx, pauli))

    return repacked
