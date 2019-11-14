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
import datetime
import heapq
import os
import time

import numpy as np

import matplotlib.pyplot as plt
import msgpack
import seaborn as sns

from gaze_mappers import GAZE_MAPPERS

sns.set_context("notebook")
sns.set_style("whitegrid")

FIGSIZE = (16, 10)
MIN_CONF = 0.6


def infinite_counter():
    i = 0
    while True:
        yield i
        i += 1


def take(n, it):
    counter = infinite_counter() if n is None else range(n)
    return [x for x, _ in zip(it, counter)]


def timestamp_from_datetime(dt):
    return time.mktime(dt.timetuple())


def timestamp_now():
    dt = datetime.datetime.now()
    return timestamp_from_datetime(dt)


def generate_timestamps(fps, start_timestamp=None):
    current_timestamp = start_timestamp or timestamp_now()

    while True:
        yield current_timestamp
        current_timestamp += 1 / fps


def generate_uniform_numbers(low=0.0, high=1.0):
    while True:
        yield np.random.uniform(low=low, high=high)


def generate_monocular_pupil_data(eye_id, confidences, timestamps):
    while True:
        try:
            yield {
                "id": eye_id,
                "confidence": next(confidences),
                "timestamp": next(timestamps),
            }
        except StopIteration:
            return


def custom_binocular_mixer(
    sample_count,
    both_camera_jitter_range=None,
    eye0_camera_jitter_range=None,
    eye1_camera_jitter_range=None,
    both_transport_jitter_range=None,
    eye0_transport_jitter_range=None,
    eye1_transport_jitter_range=None,
    both_transport_delay=None,
    eye0_transport_delay=None,
    eye1_transport_delay=None,
):
    """
    NOTE: jitter ranges should be bound to [0, 1/(2*FPS)]
    """
    eye0_cjr = eye0_camera_jitter_range or both_camera_jitter_range
    eye1_cjr = eye1_camera_jitter_range or both_camera_jitter_range

    eye0_tjr = eye0_transport_jitter_range or both_transport_jitter_range
    eye1_tjr = eye1_transport_jitter_range or both_transport_jitter_range

    eye0_td = eye0_transport_delay or both_transport_delay
    eye1_td = eye1_transport_delay or both_transport_delay

    def update_datum(
        datum, camera_jitter_range, transport_jitter_range, transport_delay
    ):
        if camera_jitter_range:
            low, high = camera_jitter_range
            datum["timestamp"] += np.random.uniform(low=low, high=high)

        arrival_time = datum["timestamp"]

        if transport_jitter_range:
            low, high = transport_jitter_range
            arrival_time += np.random.uniform(low=low, high=high)

        if transport_delay:
            arrival_time += transport_delay

        datum["arrival_time"] = arrival_time
        return datum

    def sort_key(datum):
        return datum["arrival_time"]

    def f(eye0_stream, eye1_stream):
        eye0_stream = (
            update_datum(d, eye0_cjr, eye0_tjr, eye0_td) for d in eye0_stream
        )
        eye1_stream = (
            update_datum(d, eye1_cjr, eye1_tjr, eye1_td) for d in eye1_stream
        )

        yield from heapq.merge(eye0_stream, eye1_stream, key=sort_key)

    return f


def prepare_data_for_plotting(mapped_gaze_data, segment_fn):
    segments, points = [], []

    def f(acc, x):
        acc[x["id"]] = x
        return acc

    start_time = 0  # mapped_gaze_data[0]["timestamp"]
    for datum in mapped_gaze_data:
        datum["timestamp"] -= start_time
        base_data = {}
        for d in datum["base_data"]:
            base_data[d["id"]] = d

        eye0 = base_data.get(0, None)
        eye1 = base_data.get(1, None)

        if eye0 and eye1:
            points.append(point(0, eye0))
            points.append(point(1, eye1))
            segments.append(segment_fn(eye0, eye1))
        elif eye0:
            points.append(point(0, eye0))
        elif eye1:
            points.append(point(1, eye1))

    return segments, points


def point(eye_id, eye_data):
    x = eye_data["timestamp"]
    y = eye_id
    size = eye_data["confidence"]
    try:
        arrival = eye_data["arrival_time"]
    except KeyError:
        print(eye_data)
    return (eye_id, size, x, y, arrival)


def segment_ts_eyeid(eye0, eye1):
    _, _, x1, y1, _ = point(0, eye0)
    _, _, x2, y2, _ = point(1, eye1)
    return (x1, y1), (x2, y2)


def segment_ts_arrival(eye0, eye1):
    _, _, x1, _, y1 = point(0, eye0)
    _, _, x2, _, y2 = point(1, eye1)
    return (x1, y1), (x2, y2)


def plot_mapped_gaze_data(mapped_gaze_data, title):
    x_low = mapped_gaze_data[0]["timestamp"]
    x_up = mapped_gaze_data[-1]["timestamp"]
    time_difference = x_up - x_low
    padding = time_difference * 0.05
    xlims = (x_low - padding, x_up + padding)

    fig, ax = plt.subplots(3, 1, figsize=FIGSIZE)
    _plot_connections_over_time(mapped_gaze_data, title, ax[0])
    _plot_ts_vs_arrival(mapped_gaze_data, title, ax[1])
    _plot_gaze_ts_diff(mapped_gaze_data, title, ax[2])

    ax[0].set_xlim(xlims)
    ax[1].set_xlim(xlims)
    ax[2].set_xlim(xlims)

    plt.tight_layout()


def _plot_connections_over_time(mapped_gaze_data, title, ax):
    segments, points = prepare_data_for_plotting(
        mapped_gaze_data, segment_fn=segment_ts_eyeid
    )
    ax.set_title("Binocular matches by time:" + title)
    for (x1, y1), (x2, y2) in segments:
        connection = ax.plot((x1, x2), (y1, y2), color="red", linestyle="solid")

    eye_id, size, x, y, arrival = zip(*points)

    min_markersize = 50
    max_markersize = 200
    confidence = np.asanyarray(size)
    markersize = np.where(confidence < MIN_CONF, min_markersize, max_markersize)

    ax.scatter(x, y, c=arrival, marker="o", s=markersize, cmap="viridis")

    ax.set_ylabel("Eye ID")
    ax.set_yticks([0, 1])

    ax.set_xlabel("Timestamps [seconds]")
    ax.set_xlim(x[0], x[-1])

    cutoff_x = [gp["timestamp"] for gp in mapped_gaze_data if "temporal_cutoff" in gp]
    cutoff_y = [
        gp["temporal_cutoff"] for gp in mapped_gaze_data if "temporal_cutoff" in gp
    ]

    ax_right = ax.twinx()
    ax_right.plot(cutoff_x, cutoff_y, color="green")
    ax_right.set_ylabel("Adaptive Temporal Cutoff [seconds]")


def _plot_ts_vs_arrival(mapped_gaze_data, title, ax):
    segments, points = prepare_data_for_plotting(
        mapped_gaze_data, segment_fn=segment_ts_arrival
    )
    ax.set_title("Creation time $vs$ arrival time:" + title)

    for (x1, y1), (x2, y2) in segments:
        connection = ax.plot((x1, x2), (y1, y2), color="blue", linestyle="solid")

    eye_id, size, x, y, arrival = zip(*points)

    min_markersize = 50
    max_markersize = 200
    confidence = np.asanyarray(size)
    markersize = np.where(confidence < MIN_CONF, min_markersize, max_markersize)

    ax.scatter(x, arrival, marker="o", s=markersize, c=y, cmap="Set1")

    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Timestamps [seconds]")
    ax.set_ylabel("Arrival time [seconds]")


def _plot_gaze_ts_diff(mapped_gaze_data, title, ax):
    ax.set_title("Timestamp difference by stream:" + title)

    streams = collections.defaultdict(list)
    for datum in mapped_gaze_data:
        base_data = datum["base_data"]
        if len(base_data) == 2:
            topic = "binocular"
        elif len(base_data) == 1:
            topic = "eye" + str(base_data[0]["id"])
        else:
            topic = "no_data"
        streams[topic].append(datum["timestamp"])

    for topic, ts in streams.items():
        ax.plot(ts[1:], np.diff(ts), "+-", label=topic)

    ax.legend()

    ax.set_xlabel("Order of mapping")
    ax.set_ylabel("Timestamp difference")


def generate_mapped_gaze_data_samples(
    sample_count,
    gaze_mappers,
    binocular_data_mixer,
    eye0_confidence_stream,
    eye1_confidence_stream,
    eye0_timestamp_fps,
    eye1_timestamp_fps,
    eye0_timestamp_start=None,
    eye1_timestamp_start=None,
):
    assert len(gaze_mappers) > 0

    result = []

    eye0_timestamps = take(
        sample_count,
        generate_timestamps(
            fps=eye0_timestamp_fps, start_timestamp=eye0_timestamp_start,
        ),
    )

    eye1_timestamps = take(
        sample_count,
        generate_timestamps(
            fps=eye1_timestamp_fps, start_timestamp=eye1_timestamp_start,
        ),
    )

    eye0_confidences = take(sample_count, eye0_confidence_stream)

    eye1_confidences = take(sample_count, eye1_confidence_stream)

    for gaze_mapper in gaze_mappers:

        eye0_stream = generate_monocular_pupil_data(
            eye_id=0,
            timestamps=iter(eye0_timestamps),
            confidences=iter(eye0_confidences),
        )

        eye1_stream = generate_monocular_pupil_data(
            eye_id=1,
            timestamps=iter(eye1_timestamps),
            confidences=iter(eye1_confidences),
        )

        pupil_data_stream = binocular_data_mixer(eye0_stream, eye1_stream)
        pupil_data = take(None, pupil_data_stream)

        mapped_data = gaze_mapper.map_batch(pupil_data)

        result.append(mapped_data)

    return result


def evaluate_gaze_mappers(
    binocular_data_mixer,
    eye0_confidence_stream,
    eye1_confidence_stream,
    sample_count=60,
    both_timestamp_fps=None,
    eye0_timestamp_fps=None,
    eye1_timestamp_fps=None,
    both_timestamp_start=timestamp_now(),
    eye0_timestamp_start=None,
    eye1_timestamp_start=None,
):
    eye0_timestamp_fps = eye0_timestamp_fps or both_timestamp_fps
    eye1_timestamp_fps = eye1_timestamp_fps or both_timestamp_fps

    assert eye0_timestamp_fps is not None
    assert eye1_timestamp_fps is not None

    eye0_timestamp_start = eye0_timestamp_start or both_timestamp_start
    eye1_timestamp_start = eye1_timestamp_start or both_timestamp_start

    assert eye0_timestamp_start is not None
    assert eye1_timestamp_start is not None

    gaze_mapped_data = generate_mapped_gaze_data_samples(
        sample_count=sample_count,
        gaze_mappers=GAZE_MAPPERS,
        binocular_data_mixer=binocular_data_mixer,
        eye0_confidence_stream=generate_uniform_numbers(),
        eye1_confidence_stream=generate_uniform_numbers(),
        eye0_timestamp_fps=eye0_timestamp_fps or shared_timestamp_fps,
        eye1_timestamp_fps=eye1_timestamp_fps or shared_timestamp_fps,
        eye0_timestamp_start=eye0_timestamp_start or shared_timestamp_start,
        eye1_timestamp_start=eye1_timestamp_start or shared_timestamp_start,
    )

    for mapper, mapped_data in zip(GAZE_MAPPERS, gaze_mapped_data):
        mapper_name = type(mapper).__name__
        print(mapper_name)
        plot_mapped_gaze_data(mapped_data, title=f"Mapper: {mapper_name}")
        plt.show()


#


PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])


def load_pldata_file(directory, topic):
    """Modified version of
    https://github.com/pupil-labs/pupil/blob/...
    cf8b845d4491d0a845be77fbc5584277a3210e16/...
    pupil_src/shared_modules/file_methods.py#L137
    """
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    msgpack_file = os.path.join(directory, topic + ".pldata")
    data = []
    topics = []
    data_ts = []
    with open(msgpack_file, "rb") as fh:
        for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):
            deserialised = msgpack.unpackb(payload, raw=False, use_list=False)
            data.append(deserialised)
            topics.append(topic)
    return PLData(data, data_ts, topics)
