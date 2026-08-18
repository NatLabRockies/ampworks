"""
Functions for adding derived columns to a :class:`~ampworks.Dataset`. Functions
always returns a modified copy rather than mutating the input, so the original
stays intact.

All default arguments assume datasets have already been standardized using
:func:`~ampworks.standardize_headers`, so most functions work with no extra
arguments. However, alias parameters are available to override column names
when needed.
    
"""
from ._labels import (
    StepLabel,
    SegmentLabel,
    add_step_labels,
    add_segment_labels,
    add_state,
    add_control_mode,
)

__all__ = [
    'StepLabel',
    'SegmentLabel',
    'add_step_labels',
    'add_segment_labels',
    'add_state',
    'add_control_mode',
]