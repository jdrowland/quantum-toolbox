"""Sorted insertion grouping algorithm with optimizations."""

from typing import Optional, List, Tuple
from qtoolbox.core.hamiltonian import Hamiltonian
from qtoolbox.core.group import GroupCollection, PauliGroup
from qtoolbox.core.pauli import PauliString


def sorted_insertion_grouping(
    hamiltonian: Hamiltonian,
    k: Optional[int] = None,
    sort_descending: bool = True,
    use_fast_path: bool = True
) -> GroupCollection:
    """Group Pauli terms using optimized sorted insertion algorithm."""
    if use_fast_path:
        return _sorted_insertion_optimized(hamiltonian, k, sort_descending)
    else:
        return _sorted_insertion_naive(hamiltonian, k, sort_descending)


def _sorted_insertion_naive(
    hamiltonian: Hamiltonian,
    k: Optional[int],
    sort_descending: bool
) -> GroupCollection:
    """Naive O(n²) implementation - kept for validation."""
    # Sort terms by coefficient magnitude
    if sort_descending:
        sorted_ham = hamiltonian.sort_by_coefficient(descending=True)
        terms = sorted_ham.terms
    else:
        terms = hamiltonian.terms

    groups = GroupCollection()

    for term in terms:
        # Try to add to existing group
        placed = False
        for group in groups.groups:
            if can_add_to_group(term, group, k):
                group.add(term)
                placed = True
                break

        # Create new group if needed
        if not placed:
            new_group = PauliGroup()
            new_group.add(term)
            groups.groups.append(new_group)

    return groups


def _sorted_insertion_optimized(
    hamiltonian: Hamiltonian,
    k: Optional[int],
    sort_descending: bool
) -> GroupCollection:
    """Optimized O(n·m) implementation using group representatives."""
    # Sort terms by coefficient magnitude
    if sort_descending:
        sorted_ham = hamiltonian.sort_by_coefficient(descending=True)
        terms = sorted_ham.terms
    else:
        terms = hamiltonian.terms

    n_qubits = hamiltonian.num_qubits()

    # Prepare block structure for k-commuting
    if k is None:
        # Full commutation - treat as single block
        blocks = [list(range(n_qubits))]
        is_full_commute = True
    else:
        # k-commuting blocks
        n_blocks = (n_qubits + k - 1) // k
        blocks = []
        for block_idx in range(n_blocks):
            start = block_idx * k
            end = min(start + k, n_qubits)
            blocks.append(list(range(start, end)))
        is_full_commute = (len(blocks) == 1)

    # Pre-compute block masks for efficiency
    block_masks = []
    for block in blocks:
        mask = 0
        for qubit_idx in block:
            mask |= (1 << qubit_idx)
        block_masks.append(mask)

    # All valid bits mask
    all_bits_valid = (1 << n_qubits) - 1

    # Group data: list of (paulis, gx, gz, valid)
    # where gx/gz are bitwise OR of all x_bits/z_bits in group
    # and valid tracks which qubits are "valid" for fast check
    group_data: List[Tuple[List[PauliString], int, int, int]] = []

    # Process each term
    for term in terms:
        x = term.x_bits
        z = term.z_bits
        support = x | z

        placed = False

        # Try each existing group
        for group_info in group_data:
            paulis, gx, gz, valid = group_info

            # Fast path: check anticommutation against group representatives
            anticommute = (x & gz) ^ (z & gx)
            invalid_support = support & ~valid

            # If no invalid support and anticommute bits are all in invalid region, can add
            if invalid_support == 0 and (anticommute & valid) == 0:
                paulis.append(term)
                # Update group representatives
                group_info[1] |= x  # gx
                group_info[2] |= z  # gz
                placed = True
                break

            # Slow path: need to check against all terms in group
            all_commute = True
            for other in paulis:
                x2, z2 = other.x_bits, other.z_bits
                anticommute_full = (x & z2) ^ (z & x2)

                if anticommute_full == 0:
                    # Commute everywhere
                    continue

                # Check k-commutation: must commute within each block
                if is_full_commute:
                    # Single block - just check if even number of anticommutations
                    if bin(anticommute_full & block_masks[0]).count('1') % 2 != 0:
                        all_commute = False
                        break
                else:
                    # Multiple blocks - check each independently
                    for mask in block_masks:
                        if bin(anticommute_full & mask).count('1') % 2 != 0:
                            all_commute = False
                            break
                    if not all_commute:
                        break

            if all_commute:
                paulis.append(term)
                # Update representatives with conservative valid region
                group_info[1] |= x  # gx
                group_info[2] |= z  # gz
                group_info[3] = valid & ~anticommute  # valid
                placed = True
                break

        # Create new group if not placed
        if not placed:
            group_data.append([[term], x, z, all_bits_valid])

    # Convert to GroupCollection
    groups = GroupCollection()
    for paulis, _, _, _ in group_data:
        group = PauliGroup()
        for pauli in paulis:
            group.add(pauli)
        groups.groups.append(group)

    return groups


def can_add_to_group(term: PauliString, group: PauliGroup, k: Optional[int]) -> bool:
    """Check if term can be added to group (naive version)."""
    if k is None:
        # Full commutation required
        return all(term.commutes_with(p) for p in group.paulis)
    else:
        # k-commutation required
        return all(term.k_commutes(p, k) for p in group.paulis)
