"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# Implementation taken from https://github.com/pupil-labs/pupil/pull/1557
# https://github.com/cboulay/pupil/tree/db16290fe1b7f5b8d0bc9a19d24b948b2c8aadfb

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


class Monocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""

    def __init__(self, g_pool):
        super().__init__(g_pool)

    def on_pupil_datum(self, p):
        g = self._map_monocular(p)
        if g:
            return [g]
        else:
            return []


class Binocular_Gaze_Mapper_Base(Gaze_Mapping_Plugin):
    """Base class to implement the map callback"""

    def __init__(
        self,
        g_pool,
        min_pupil_confidence=0.6,
        temporal_cutoff=0.03,
        timestamp_reset_threshold=5.0,
        process_low_confidence=True,
        sample_cutoff=10,
    ):
        super().__init__(g_pool)
        self.min_pupil_confidence = min_pupil_confidence
        self.temporal_cutoff = temporal_cutoff
        self.timestamp_reset_threshold = timestamp_reset_threshold
        self.process_low_confidence = process_low_confidence
        self.sample_cutoff = sample_cutoff
        self._cache = deque()

    def _reset(self):
        self._cache.clear()

    def map_batch(self, pupil_list):
        backup_cache = self._cache
        self._cache = deque()
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))
        self._cache = backup_cache
        return results

    def on_pupil_datum(self, p):
        """
        Process pupil datum, combine with data in cache if possible, and return gaze datum.
        :param p: pupil datum (message with topic 'pupil.0' or 'pupil.1')
        :return: List of gaze_data or [].

        Note the mapping logic has recently changed. See htt§ps://github.com/pupil-labs/pupil/pull/1557

        One objective of the new logic is to never push a sample with an older timestamp than a sample that has already
        been pushed. This is not simple to achieve because sometimes a new sample will arrive (e.g. from eye B) with an
        older timestamp than a previous sample (e.g. from eye A). i.e. A0-A1-A2-B0-A3-B1-A4-B2-A5 etc.
        If we wish to combine samples from the two eyes for binocular reconstruction then we cannot push samples
        immediately upon receipt and we must maintain a buffer of samples. How to do that without incurring too much
        delay is tricky.

        The only thing we can be sure of is that a new sample from one eye will have a later timestamp than
        previous samples from that same eye.
        """
        # Reset state if we receive a new timestamp that is too old (probably due to clock reset).
        if len(self._cache) > 0:
            t_delta = np.abs(p["timestamp"] - self._cache[-1]["timestamp"])
            if t_delta > self.timestamp_reset_threshold:
                self._reset()

        # Insert the new sample in the cache according to its timestamp.
        if (len(self._cache) == 0) or (p["timestamp"] > self._cache[-1]["timestamp"]):
            # Shortcut when timestamp is newer than anything in cache.
            self._cache.append(p)
        else:
            cache_ts = [_["timestamp"] for _ in self._cache]
            in_ix = np.searchsorted(cache_ts, p["timestamp"])
            self._cache.insert(in_ix, p)

        # Process the cache left to right, stopping when we cannot confidently pop a sample.
        output = []
        for ix in range(len(self._cache) - 1):
            # We can only ever pop a sample when there is a newer sample from the opposite eye,
            # because then we know there cannot be a sample inserted older than the current sample.
            newer_other = -1
            do_binoc = False

            for ix, other_samp in enumerate(self._cache):
                if other_samp["id"] == 1 - self._cache[0]["id"]:
                    newer_other = ix

                    # If current sample is low-conf then we don't need to search anymore. _Any_ other-eye samp is OK.
                    if not self._cache[0]["confidence"]:
                        break

                    # If newer_other is more than temporal_cutoff newer then we don't need to search anymore.
                    time_diff = other_samp["timestamp"] - self._cache[0]["timestamp"]
                    if time_diff > self.temporal_cutoff:
                        break

                    # We know from above that current sample is high-conf and newer_other is within temporal_cutoff.
                    # If newer_other is also high-confidence then we can do binocular mapping.
                    if other_samp["confidence"] >= self.min_pupil_confidence:
                        do_binoc = True
                        break

            # If we have failed to find a newer sample from the other eye
            # then we cannot do anything because the other eye might yet insert a sample older than the current sample.
            # The only exception is if the cache is overflowing, then do _map_monocular anyway.
            if (newer_other < 0) and (len(self._cache) <= self.sample_cutoff):
                break

            if do_binoc:
                if self._cache[0]["id"] == 0:
                    mapped_sample = self._map_binocular(
                        self._cache[0], self._cache[newer_other]
                    )
                else:
                    mapped_sample = self._map_binocular(
                        self._cache[newer_other], self._cache[0]
                    )
            else:
                mapped_sample = self._map_monocular(self._cache[0])
            output.append(mapped_sample)

            # Pop the first sample off our cache
            self._cache.popleft()

            # Cannot continue unless we have more than 1 sample in the cache.
            if len(self._cache) == 1:
                break

        return output
