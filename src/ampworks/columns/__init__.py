"""
Functions for adding derived columns to a :class:`~ampworks.Dataset`. Each
function returns a modified copy rather than mutating the input. Default
column names assume :class:`~ampworks.HeaderAliases` standardization, but can
be overridden via alias arguments.

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
