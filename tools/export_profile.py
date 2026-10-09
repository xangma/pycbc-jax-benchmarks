#!/usr/bin/env python3
"""Export portable, allowlisted intervals from a private Nsight SQLite capture.

The supplied verified summary defines analysis windows and NVTX clock bounds.
Only selected application kernel/memcpy events, named host phases and JAX
monitoring points are exported. No raw input, runtime path, PID, StringIds table,
launch command or arbitrary kernel label is published. Event boundaries remain
integer nanoseconds relative to the original session clock.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3

PHASES = set("startup_imports public_search public_live search_batch segment_fft "
             "conditioning_and_frame_io psd template_decompression template_norms "
             "matched_filter_cluster point_chisq event_finalize hdf_output search "
             "search_loop live_loop steady_loop warm warmed_compute warm_search "
             "warm_loop steady".split())
NATIVE = set("block output postprocess coinc_best_publication_readback coinc_query_host_prepare "
             "coinc_resident_admitted coinc_resident_cluster_dispatch "
             "coinc_resident_compatibility coinc_resident_publication_readback "
             "coinc_resident_query_dispatch coinc_singles_group_append_dispatch "
             "coinc_singles_group_host_prepare coinc_singles_group_index_dispatch "
             "coincidence_dispatch frame_read gwf_cuda_decompress gwf_parse "
             "gwf_replay_buffer mpi_ack_send mpi_drain mpi_gather mpi_receive "
             "mpi_unpickle output_drain output_readback output_snapshot output_write "
             "psd_refresh_batch device_to_host filter filter_bank_length_check "
             "filter_bank_setup filter_bank_sort filter_fused filter_norms "
             "filter_overwhiten filter_veto filter_veto_bins filter_veto_bins_collect "
             "filter_veto_bins_dispatch filter_veto_bins_prepare "
             "filter_veto_executable_load filter_veto_executable_lower "
             "filter_waveform_generation filter_workspace_setup mpi_ack_wait "
             "mpi_post mpi_send_completion mpi_serialize mpi_snapshot_wait".split())
MONITOR = {
    "/jax/compilation_cache/cache_hits", "/jax/compilation_cache/cache_misses",
    "/jax/compilation_cache/compile_requests_use_cache",
    "/jax/compilation_cache/tasks_using_cache",
    "/jax/core/compile/backend_compile_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/jaxpr_trace_duration",
}
COPY = {1: "H2D", 2: "D2H", 8: "D2D"}


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def export(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("The source revision must be a full lowercase Git SHA")
    reference = json.loads(args.summary.read_text())
    origin = int(reference["analysis_origin_system_clock_ns"])
    database = reference["databases"][0]
    # Only these audited process identities are admitted; neither PID form leaves
    # the exporter. GPU events below must belong to these application reports.
    reports = [json.loads(path.read_text()) for path in args.report]
    pid_labels = {int(r["pid"]): f"rank {int(r['rank'])}" if "rank" in r else "search"
                  for r in reports if "pid" in r and
                  ("rank" in r or (args.app == "inspiral" and "spans" in r))}
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    admitted = {int(x["global_pid"]) for x in database["selected_processes"]}
    processes = {int(gpid): pid_labels[int(pid)] for gpid, pid in connection.execute(
        "SELECT globalPid,pid FROM PROCESSES") if int(gpid) in admitted and int(pid) in pid_labels}
    if not processes:
        raise ValueError("No selected application process matched its observer receipt")
    selected = ",".join(str(pid) for pid in sorted(processes))
    gpu = []
    actual_processes = set()
    for table, direction in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                             ("CUPTI_ACTIVITY_KIND_MEMCPY", None)):
        fields = "start,end,globalPid,deviceId"
        fields += ",copyKind,bytes" if direction is None else ",0,0"
        for lo, hi, gpid, device, copy_kind, byte_count in connection.execute(
                f"SELECT {fields} FROM {table} WHERE globalPid IN ({selected}) AND end>start"):
            label = direction or COPY[copy_kind]
            gpu.append([int(lo), int(hi), processes[int(gpid)], int(device), label, int(byte_count)])
            actual_processes.add(processes[int(gpid)])
    if actual_processes != set(pid_labels.values()):
        raise ValueError("Some selected application processes have no GPU events")
    gpu.sort(key=lambda row: row[0])
    host = []
    normalization = database["nvtx_clock_normalization"]
    if normalization["status"] != "passed":
        raise ValueError("Verified NVTX clock normalization is required")
    upper = int(normalization["bounds"]["end_ns"]) - origin
    nvtx_names = set()
    for lo, hi, text, gpid in connection.execute(
            "SELECT start,end,COALESCE(text,(SELECT value FROM StringIds WHERE id=textId)),"
            "(globalTid & -16777216) FROM NVTX_EVENTS "
            f"WHERE (globalTid & -16777216) IN ({selected}) "
            "AND eventType IN (59,60) AND end>start"):
        try:
            value = json.loads(text)
            name = value.get("name", value.get("function")) if isinstance(value, dict) else text
        except (TypeError, ValueError):
            name = text
        if name not in PHASES:
            continue
        candidates = [(int(lo), int(hi))] if 0 <= lo < hi <= upper else []
        if origin <= lo < hi <= origin + upper:
            candidates.append((int(lo)-origin, int(hi)-origin))
        if len(candidates) != 1:
            raise ValueError("Named NVTX phase has ambiguous or invalid clock domain")
        begin, end = candidates[0]
        host.append([begin, end, processes[int(gpid)], name, "NVTX"])
        nvtx_names.add(name)
    for report in reports:
        pid = report.get("pid")
        if pid not in pid_labels:
            continue
        process = pid_labels[pid]
        for obj in walk(report):
            name = obj.get("name", obj.get("phase", obj.get("label")))
            if name not in PHASES or name in nvtx_names:
                continue
            if "start_ns" in obj and "end_ns" in obj:
                lo, hi = int(obj["start_ns"]), int(obj["end_ns"])
                if hi > lo:
                    host.append([lo-origin, hi-origin, process, name, "wrapper"])
        if "rank" in report:
            lo, hi = int(report["cli_start_ns"]), int(report["completed_ns"])
            host.append([lo-origin, hi-origin, process, "public_live", "wrapper completion fence"])
            if report.get("blocks"):
                host.append([lo-origin, min(int(x["start_ns"]) for x in report["blocks"])-origin,
                             process, "startup/setup", "wrapper"])
            for timestamp, name in report.get("compile_events", []):
                if name not in MONITOR:
                    raise ValueError("Unrecognized JAX monitoring event; extend the explicit allowlist")
                t = int(timestamp)-origin
                host.append([t, t, process, name, "monitor point"])
    pending = collections.defaultdict(list)
    for path in args.stages:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line.removeprefix("PYCBC_STAGE_EVENT "))
            if int(event["pid"]) not in pid_labels:
                continue
            name = event["stage"]
            if name not in NATIVE:
                raise ValueError(f"Unrecognized native phase: {name!r}")
            key = (int(event["pid"]), name, event.get("data_start"), event.get("data_end"))
            if event["event"] == "start":
                pending[key].append(event)
            elif event["event"] == "end" and pending[key]:
                begin = pending[key].pop()
                lo, hi = int(begin["monotonic_ns"])-origin, int(event["monotonic_ns"])-origin
                if hi > lo:
                    host.append([lo, hi, pid_labels[key[0]], name, "native host stage"])
    host.sort(key=lambda row: row[0])
    basis = {
        "whole": "Extent of selected application recorded GPU activity and host ranges; full launch-to-exit timing is separate.",
        "search": ("First through last measured search_batch host range; includes intervening work, without a device fence."
                   if args.app == "inspiral" else
                   "Combined public_live host scope from CLI entry through final completion fence."),
        "warm": ("Repeated middle search batch; absence of compilation is unverified."
                 if args.app == "inspiral" else
                 "Common valid-data suffix after two valid blocks through final completion fence; zero-compilation unverified."),
        "detail": "Deterministic central detail window of the suffix; same unverified warm status.",
    }
    windows = {}
    for name, value in reference["windows"].items():
        lo, hi = round(value["start_seconds"]*1e9), round(value["end_seconds"]*1e9)
        # Float text is used only for these externally verified boundaries; GPU
        # and host events retain original integer nanoseconds without rounding.
        expected_union = {key: round(value["recorded_gpu_activity_union_seconds"][key]*1e9)
                          for key in ("all", "kernel", "copy")}
        expected_bytes = collections.Counter()
        for row in value["completed_transfer_bytes"]:
            expected_bytes[(pid_labels[int(row["pid"])], int(row["device"]), row["direction"])] += row["bytes"]
        windows[name] = {
            "start_ns": lo, "end_ns": hi, "warm_status_verified": value["warm_status_verified"],
            "boundary_basis": basis[name], "expected_union_ns": expected_union,
            "expected_event_counts": value["overlapping_event_counts"],
            "expected_completed_transfer_bytes": [[*key, count] for key, count in sorted(expected_bytes.items())],
        }
    result = {
        "format": "pycbc-jax-profile-intervals-v1", "app": args.app,
        "source_revision": args.revision, "time_unit": "integer nanoseconds from session start",
        "processes": sorted(actual_processes), "windows": windows,
        "gpu_columns": ["start_ns", "end_ns", "process", "device", "direction", "bytes"],
        "host_columns": ["start_ns", "end_ns", "process", "phase", "source"],
        "gpu": gpu, "host": host,
        "unpaired_native_stage_starts_excluded": sum(map(len, pending.values())),
        "private_input_sha256": {"sqlite": sha256(args.sqlite), "verified_summary": sha256(args.summary)},
        "definitions": {
            "gpu": "Selected application kernel/memcpy intervals only; memsets excluded. Union measures recorded activity, not occupancy or whole-device utilization.",
            "host": "Allowlisted native/observer host submission phases; no implied device fence, except explicitly named final completion scopes.",
            "monitor": "Callback point timestamps, not durations. Backend notifications include cache retrieval and do not establish cold compilation.",
            "export": "PID identities replaced by application/rank labels. Times and durations are unchanged; no arbitrary StringIds, kernel labels, commands, paths or input strain are exported.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Empty gzip filename and fixed mtime make the same export byte reproducible.
    with args.output.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write(json.dumps(result, separators=(",", ":"), sort_keys=True).encode())
    connection.close()
    print(json.dumps({"app": args.app, "gpu_events": len(gpu), "host_events": len(host),
                      "bytes": args.output.stat().st_size, "sha256": sha256(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=("inspiral", "live"), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--stages", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
