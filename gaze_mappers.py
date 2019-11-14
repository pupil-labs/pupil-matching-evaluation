"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import collections

import numpy as np

from pr_1557_monotonic_gaze_mappers import (
    Binocular_Gaze_Mapper_Base as Monotonic_Gaze_Mapper_Base,
)
from pr_1728_deduplicated_gaze_mappers import (
    Binocular_Gaze_Mapper_Base as No_Upsampling_Gaze_Mapper_Base,
)
from pr_1731_dynamic_cutoff_gaze_mappers import (
    Binocular_Gaze_Mapper_Base as Dynamic_Cutoff_Gaze_Mapper_Base,
)
from v1_18_gaze_mappers import (
    Binocular_Gaze_Mapper_Base as v1_18_Binocular_Gaze_Mapper_Base,
)


class Binocular_Gaze_Mapper_Mixin:
    def __init__(
        self,
        min_pupil_confidence=0.6,
        temporal_cutoff=0.3,
        sample_cutoff=10,
        timestamp_reset_threshold=5.0,
        current_cutoff=1 / 120,
        temporal_cutoff_smoothing_ratio=1 / 50,
    ):
        self.min_pupil_confidence = min_pupil_confidence
        self._caches = (collections.deque(), collections.deque())
        self.temporal_cutoff = temporal_cutoff
        self.sample_cutoff = sample_cutoff

        # PR #
        self._cache = collections.deque()
        self.timestamp_reset_threshold = timestamp_reset_threshold

        # dynamic cutoff
        self.current_cutoff = current_cutoff
        self.temporal_cutoff_smoothing_ratio = temporal_cutoff_smoothing_ratio

    def _map_binocular(self, p0, p1, temporal_cutoff=0.0):
        return {
            "confidence": np.mean([p0["confidence"], p1["confidence"]]),
            "timestamp": np.mean([p0["timestamp"], p1["timestamp"]]),
            "base_data": [p0, p1],
            "temporal_cutoff": temporal_cutoff,
        }

    def _map_monocular(self, p, temporal_cutoff=0.0):
        return {
            "confidence": p["confidence"],
            "timestamp": p["timestamp"],
            "base_data": [p],
            "temporal_cutoff": temporal_cutoff,
        }


class v1_18_Binocular_Gaze_Mapper(
    Binocular_Gaze_Mapper_Mixin, v1_18_Binocular_Gaze_Mapper_Base
):
    pass


class Monotonic_Binocular_Gaze_Mapper(
    Binocular_Gaze_Mapper_Mixin, Monotonic_Gaze_Mapper_Base
):
    pass


class No_Upsampling_Binocular_Gaze_Mapper(
    Binocular_Gaze_Mapper_Mixin, No_Upsampling_Gaze_Mapper_Base
):
    pass


class Dynamic_Cutoff_Binocular_Gaze_Mapper(
    Binocular_Gaze_Mapper_Mixin, Dynamic_Cutoff_Gaze_Mapper_Base
):
    pass


GAZE_MAPPERS = (
    v1_18_Binocular_Gaze_Mapper(),
    Monotonic_Binocular_Gaze_Mapper(),
    No_Upsampling_Binocular_Gaze_Mapper(),
    Dynamic_Cutoff_Binocular_Gaze_Mapper(),
)
