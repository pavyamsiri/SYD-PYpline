"""
NamedTuple representing target data that can be processed by the SYD-PYpline routines.
"""

from typing import NamedTuple

import numpy as np


class TargetData(NamedTuple):
    # File information
    target: str
    path: str
    # Time series data
    lc_time: np.ndarray
    lc_flux: np.ndarray
    cadence: float
    nyquist: float
    # Power spectrum data
    ps_frequency: np.ndarray
    ps_power: np.ndarray
    oversample: float
    resolution: float
