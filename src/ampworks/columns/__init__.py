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
from ._rates import (
    add_power,
    add_c_rate,
)
from ._sequencing import (
    add_instance_nums,
    add_relative_time,
)
from ._capacity import (
    add_capacity,
    add_cumulative_capacity,
    add_throughput_capacity,
    add_equivalent_full_cycles,
)
from ._energy import (
    add_energy,
    add_cumulative_energy,
    add_throughput_energy,
)

__all__ = [
    'StepLabel',
    'SegmentLabel',
    'add_step_labels',
    'add_segment_labels',
    'add_state',
    'add_control_mode',
    'add_power',
    'add_c_rate',
    'add_instance_nums',
    'add_relative_time',
    'add_capacity',
    'add_cumulative_capacity',
    'add_throughput_capacity',
    'add_equivalent_full_cycles',
    'add_energy',
    'add_cumulative_energy',
    'add_throughput_energy',
]
