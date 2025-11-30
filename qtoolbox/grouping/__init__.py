"""Grouping algorithms for Pauli operators."""

from qtoolbox.core.group import GroupCollection
from qtoolbox.grouping.sorted_insertion import sorted_insertion_grouping
from qtoolbox.grouping.adhoc_repacking import adhoc_repacking
from qtoolbox.grouping.posthoc_repacking import posthoc_repacking

__all__ = [
    "GroupCollection",
    "sorted_insertion_grouping",
    "adhoc_repacking",
    "posthoc_repacking",
]
