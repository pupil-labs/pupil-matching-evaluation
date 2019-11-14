"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

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
        self.temporal_cutoff = 0.3
        self.sample_cutoff = 10

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

        # map low confidence pupil data monocularly
        if (
            self._caches[0]
            and self._caches[0][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[0].popleft()
            gaze_datum = self._map_monocular(p, self.temporal_cutoff)
        elif (
            self._caches[1]
            and self._caches[1][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[1].popleft()
            gaze_datum = self._map_monocular(p, self.temporal_cutoff)
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

            if abs(p0["timestamp"] - p1["timestamp"]) < self.temporal_cutoff:
                gaze_datum = self._map_binocular(p0, p1, self.temporal_cutoff)
            else:
                gaze_datum = self._map_monocular(older_pt, self.temporal_cutoff)

        elif len(self._caches[0]) > self.sample_cutoff:
            p = self._caches[0].popleft()
            gaze_datum = self._map_monocular(p, self.temporal_cutoff)
        elif len(self._caches[1]) > self.sample_cutoff:
            p = self._caches[1].popleft()
            gaze_datum = self._map_monocular(p, self.temporal_cutoff)
        else:
            gaze_datum = None

        if gaze_datum:
            return [gaze_datum]
        else:
            return []
