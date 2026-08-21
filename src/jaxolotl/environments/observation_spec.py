"""Shape and dtype specifications for structured environment observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

import numpy as np


@dataclass(frozen=True)
class ArraySpec:
    """Static shape and dtype metadata for one observation array."""

    shape: tuple[int, ...]
    dtype: np.dtype


class ObservationSpec(Mapping[str, ArraySpec]):
    def __init__(self, **kwargs: ArraySpec):
        self._specs = kwargs

    def __getitem__(self, key: str) -> ArraySpec:
        return self._specs[key]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)

    @property
    def flat_observation_size(self) -> int:
        """Return the total number of scalar values in this observation spec."""
        return sum(prod(spec.shape) for spec in self.values())
