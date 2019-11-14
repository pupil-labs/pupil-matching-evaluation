"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# Taken from https://github.com/pupil-labs/pupil/pull/1731
# https://github.com/romanroibu/pupil/tree/3395363f5b6417385775f0d9f903b2f8ae73fdb4

from collections import deque
import numpy as np


class Gaze_Mapping_Plugin:
    """base class for all gaze mapping routines"""

    uniqueness = "by_base_class"
    order = 0.1
    icon_chr = chr(0xEC20)
    icon_font = "pupil_icons"

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.g_pool.active_gaze_mapping_plugin = self

    def on_pupil_datum(self, p):
        raise NotImplementedError()

    def map_batch(self, pupil_list):
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))
        return results

    def add_menu(self):
        super().add_menu()
        self.menu_icon.order = 0.31


class Binocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.min_pupil_confidence = 0.6
        self._caches = (deque(), deque())
        self.current_cutoff = 1 / 120
        self.temporal_cutoff_smoothing_ratio = 0.35
        self.sample_cutoff = 10

    def is_cache_valid(self, cache):
        return len(cache) >= 2

    def calculate_raw_cutoff(self, cache):
        return np.mean(np.diff([d["timestamp"] for d in cache]))

    def calculate_temporal_cutoff(self, eye0_cache, eye1_cache):

        if self.is_cache_valid(eye0_cache) and self.is_cache_valid(eye1_cache):
            raw_cutoff = max(
                self.calculate_raw_cutoff(eye0_cache),
                self.calculate_raw_cutoff(eye1_cache),
            )
        elif self.is_cache_valid(eye0_cache):
            raw_cutoff = self.calculate_raw_cutoff(eye0_cache)
        elif self.is_cache_valid(eye1_cache):
            raw_cutoff = self.calculate_raw_cutoff(eye1_cache)
        else:
            return self.current_cutoff

        self.current_cutoff += (
            raw_cutoff - self.current_cutoff
        ) * self.temporal_cutoff_smoothing_ratio
        return self.current_cutoff

    def map_batch(self, pupil_list):
        current_caches = self._caches
        self._caches = (deque(), deque())
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))

        self._caches = current_caches
        return results

    def on_pupil_datum(self, p):
        self._caches[p["id"]].append(p)
        temporal_cutoff = 2 * self.calculate_temporal_cutoff(*self._caches)

        # map low confidence pupil data monocularly
        if (
            self._caches[0]
            and self._caches[0][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[0].popleft()
            gaze_datum = self._map_monocular(p, temporal_cutoff)
        elif (
            self._caches[1]
            and self._caches[1][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[1].popleft()
            gaze_datum = self._map_monocular(p, temporal_cutoff)
        # map high confidence data binocularly if available
        elif self._caches[0] and self._caches[1]:
            # we have binocular data
            if self._caches[0][0]["timestamp"] < self._caches[1][0]["timestamp"]:
                p0 = self._caches[0].popleft()
                p1 = self._caches[1][0]
                older_pt = p0
            else:
                p0 = self._caches[0][0]
                p1 = self._caches[1].popleft()
                older_pt = p1

            if abs(p0["timestamp"] - p1["timestamp"]) < temporal_cutoff:
                gaze_datum = self._map_binocular(p0, p1, temporal_cutoff)
            else:
                gaze_datum = self._map_monocular(older_pt, temporal_cutoff)

        elif len(self._caches[0]) > self.sample_cutoff:
            p = self._caches[0].popleft()
            gaze_datum = self._map_monocular(p, temporal_cutoff)
        elif len(self._caches[1]) > self.sample_cutoff:
            p = self._caches[1].popleft()
            gaze_datum = self._map_monocular(p, temporal_cutoff)
        else:
            gaze_datum = None

        if gaze_datum:
            return [gaze_datum]
        else:
            return []
