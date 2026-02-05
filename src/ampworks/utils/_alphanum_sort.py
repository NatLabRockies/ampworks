from __future__ import annotations

import re


def alphanum_sort(unsorted: list[str], reverse: bool = False) -> list[str]:
    """
    Sort a list alphanumerically.

    This sorting function ensures that numerical substrings are compared based
    on their integer values. For example, "item2" comes before "item10", unlike
    standard string sorting where "item10" would come before "item2".

    Parameters
    ----------
    unsorted : list[str]
        Original unsorted list of strings.
    reverse : bool, optional
        Flag to reverse the sorted list. The default is False.

    Returns
    -------
    sorted : list[str]
        An alphanumerically sorted list of strings.

    """
    unsorted = list(unsorted)

    def convert(txt): return int(txt) if txt.isdigit() else txt
    def alphanum(key): return [convert(c) for c in re.split('([0-9]+)', key)]

    out = sorted(unsorted, key=alphanum, reverse=reverse)

    return out
