#!/usr/bin/env python3
"""Verify and render portable profile intervals; needs only matplotlib.

Example: python tools/render_profile.py --input intervals.json.gz --output plots
The input contains no private Nsight database or Python runtime metadata.
All numeric summaries use exact integer interval unions. Dense plot lanes may be
binned for display only; that never changes event counts or summary unions.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def union_add(intervals, start, end):
    if end <= start:
        return
    if intervals and start <= intervals[-1][1]:
        intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
    else:
        intervals.append((start, end))


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        union_add(merged, start, end)
    return merged


def ns(intervals):
    return sum(end-start for start, end in intervals)


def clipped(intervals, lo, hi):
    return union((max(start, lo), min(end, hi)) for start, end in intervals
                 if end > lo and start < hi)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize(data, window):
    lo, hi = window["start_ns"], window["end_ns"]
    unions = collections.defaultdict(list)
    lanes = collections.defaultdict(list)
    counts, byte_counts = collections.Counter(), collections.Counter()
    for start, end, process, device, direction, size in data["gpu"]:
        if end <= lo or start >= hi:
            continue
        begin, finish = max(start, lo), min(end, hi)
        kind = "kernel" if direction == "kernel" else "copy"
        for key in ("all", kind):
            union_add(unions[key], begin, finish)
        union_add(lanes[(process, device, direction)], begin, finish)
        counts[kind] += 1
        if kind == "copy" and lo <= end < hi:
            counts[f"completed_{direction}"] += 1
            byte_counts[(process, device, direction)] += size
    measured = {"start_ns": lo, "end_ns": hi, "duration_ns": hi-lo,
                "recorded_gpu_activity_union_ns": {key: ns(unions[key]) for key in ("all", "kernel", "copy")},
                "recorded_gpu_activity_fraction": ns(unions["all"])/(hi-lo),
                "overlapping_event_counts": dict(counts),
                "completed_transfer_bytes": [[*key, size] for key, size in sorted(byte_counts.items())],
                "warm_status_verified": window["warm_status_verified"],
                "boundary_basis": window["boundary_basis"]}
    if measured["recorded_gpu_activity_union_ns"] != window["expected_union_ns"]:
        raise ValueError("Portable intervals disagree with verified union durations")
    if measured["overlapping_event_counts"] != window["expected_event_counts"]:
        raise ValueError("Portable intervals disagree with verified event counts")
    if measured["completed_transfer_bytes"] != window["expected_completed_transfer_bytes"]:
        raise ValueError("Portable intervals disagree with verified completed transfer bytes")
    return measured, lanes


def display_intervals(intervals, lo, hi, maximum=12000):
    if len(intervals) <= maximum:
        return intervals, False
    width = max(1, math.ceil((hi-lo)/maximum))
    bins = set()
    for start, end in intervals:
        bins.update(range((start-lo)//width, (end-1-lo)//width+1))
    return union((lo+index*width, min(hi, lo+(index+1)*width)) for index in sorted(bins)), True


def plot(data, name, measured, lanes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = measured["start_ns"], measured["end_ns"]
    hosts = collections.defaultdict(list)
    points = collections.defaultdict(list)
    for start, end, process, phase, source in data["host"]:
        if source == "monitor point":
            if lo <= start < hi:
                points[f"{process}: JAX trace/lowering/backend/cache callbacks (points)"].append(start)
        elif end > lo and start < hi:
            hosts[f"{process}: {phase}"].append((max(lo, start), min(hi, end)))
    for label in hosts:
        hosts[label] = union(hosts[label])
    order = sorted(hosts, key=lambda label: (
        0 if re.search(r"startup|import|compil|warm|public_", label) else 1,
        -ns(hosts[label]), label))
    rows = [(f"host: {label}", hosts[label], "#7c8ba1", False) for label in order]
    rows.extend((label, [(time, time) for time in points[label]], "#d28c24", True)
                for label in sorted(points))
    colors = {"kernel": "#226aa8", "H2D": "#41a75e", "D2H": "#d55353", "D2D": "#9971b5"}
    rows.extend((f"GPU {device}, {process}: {direction}", intervals, colors[direction], False)
                for (process, device, direction), intervals in sorted(lanes.items()))
    figure, axis = plt.subplots(figsize=(15, max(4.2, .28*len(rows)+2)))
    binned = False
    for index, (_, intervals, color, point) in enumerate(rows):
        if point:
            axis.vlines([start/1e9 for start, _ in intervals], index-.3, index+.3,
                        colors=color, linewidth=.7)
        else:
            intervals, did_bin = display_intervals(intervals, lo, hi)
            binned |= did_bin
            axis.broken_barh([(start/1e9, (end-start)/1e9) for start, end in intervals],
                             (index-.34, .68), facecolors=color, linewidth=0, rasterized=True)
    axis.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(lo/1e9, hi/1e9)
    axis.set_xlabel("Seconds from captured Nsight session start")
    scope = name
    if name == "warm":
        scope = "post-two-valid-block suffix" if data["app"] == "live" else "repeated batch 13/24"
    warm = "; warm status unverified" if name in ("warm", "detail") and not measured["warm_status_verified"] else ""
    title = "Inspiral" if data["app"] == "inspiral" else "Live"
    axis.set_title(f"{title}: {scope}{warm}\nRecorded kernel/copy union: "
                   f"{measured['recorded_gpu_activity_fraction']:.2%} of this scope (not occupancy)", fontsize=12)
    axis.grid(axis="x", alpha=.2)
    foot = "Host ranges measure submission without stage fences. GPU lanes merge overlapping streams; memsets excluded."
    foot += " Callback points do not establish cold compilation."
    if binned:
        foot += " Dense lanes: display bins show any overlap; summary unions remain exact."
    figure.text(.01, .012, foot, fontsize=8)
    figure.tight_layout(rect=(0, .035, 1, 1))
    target = output/f"{name}.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return {"file": target.name, "sha256": sha256(target), "dense_lanes_display_binned": binned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.input, "rt") as stream:
        data = json.load(stream)
    if data["format"] != "pycbc-jax-profile-intervals-v1":
        raise ValueError("Unsupported portable profile format")
    if any(a[0] > b[0] for a, b in zip(data["gpu"], data["gpu"][1:])):
        raise ValueError("GPU intervals must be sorted by start time")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"app": data["app"], "source_revision": data["source_revision"],
               "portable_data_sha256": sha256(args.input), "gpu_event_count": len(data["gpu"]),
               "host_event_count": len(data["host"]), "processes": data["processes"],
               "unpaired_native_stage_starts_excluded": data["unpaired_native_stage_starts_excluded"],
               "definitions": data["definitions"], "verified_against_original_summary": True,
               "windows": {}, "plots": {}}
    for name, window in data["windows"].items():
        measured, lanes = summarize(data, window)
        summary["windows"][name] = measured
        summary["plots"][name] = plot(data, name, measured, lanes, args.output)
    import matplotlib
    summary["renderer"] = {"matplotlib_version": matplotlib.__version__,
                           "display": "PNG appearance can vary with plotting/font versions; integer event counts and unions must agree exactly."}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"app": data["app"], "verified": True, "windows": {
        name: {"seconds": row["duration_ns"]/1e9,
               "gpu_union_seconds": row["recorded_gpu_activity_union_ns"]["all"]/1e9}
        for name, row in summary["windows"].items()}}, indent=2))


if __name__ == "__main__":
    main()
