# Regenerate profile plots

The public gzip files contain allowlisted application kernel/memcpy intervals,
compact host stages and selected JAX monitoring points. Process IDs are replaced
with rank labels. Arbitrary kernel/StringIds names, commands, runtime paths,
raw strain and raw Nsight reports are excluded. Integer event timestamps and
durations are unchanged relative to session start.

| Arm | Whole activity | CLI scope | Later interval | 400 ms detail |
| --- | --- | --- | --- | --- |
| Baseline | [Plot](baseline/whole.png) | [Plot](baseline/search.png) | [Plot](baseline/warm.png) | [Plot](baseline/detail.png) |
| Filter-only | [Plot](filter-only/whole.png) | [Plot](filter-only/search.png) | [Plot](filter-only/warm.png) | [Plot](filter-only/detail.png) |
| Combined | [Plot](combined/whole.png) | [Plot](combined/search.png) | [Plot](combined/warm.png) | [Plot](combined/detail.png) |

From the benchmark repository root, run:

```sh
python tools/render_profile.py --input <CAMPAIGN>/profiles/baseline/live-intervals.json.gz --output <CAMPAIGN>/profiles/baseline
```

Only Python and matplotlib are required. The recorded renderer used matplotlib
3.11.1. It recreates `whole.png`, `search.png`, `warm.png` and the central 400 ms
`detail.png`, plus `summary.json`. Image appearance can differ with font/plotting
versions; integer counts, transfer bytes and clipped interval unions must match
exactly, or rendering aborts. Dense lanes use up to 12,000 display bins when
necessary; numerical summaries never use those bins.

`search` spans both ranks' CLI entry through final completion. `whole` is the
selected host/GPU activity extent, not process launch through profiler exit.
`warm` is a filename retained by the common renderer: it denotes the suffix after
two valid blocks and is explicitly not verified compile-free. The detail window
is deterministically centered inside that suffix. Native stages have no implied
device completion fence; monitoring points are callback notifications, not
durations or proof of cold compilation. Kernel/copy unions exclude memsets and
do not establish occupancy, saturation or whole-device utilization.

The baseline exporter includes 1,393,588 GPU events and 13,845 host intervals or
points. It explicitly excludes 132 unmatched native stage starts. The
[manifest](manifest.json) binds the portable file, exact renderer/exporter, source,
input hashes and verification receipt. The independent check recomputes complete
ordered GPU event-row hashes, all four windows' counts/unions/transfer bytes and
rank/controller boundaries directly from read-only private SQLite.

To export a new private capture, first normalize/verify its Nsight and wrapper
clock domains, then use this campaign's `tools/export_profile.py` with that verified summary,
the matching rank/controller receipts, native stage logs and full source SHA.
Do not substitute reports from another run. Raw capture and observer receipts
stay private; only the allowlisted export is published. The filter-only data contains 1,395,010 GPU events and 12,267 host intervals or
points, with 132 unmatched starts excluded. Rendering, independent numerical
checks and strict saved-output comparison pass. The combined data contains 106,228 GPU events and 12,063 host intervals or
points, with 132 unmatched starts excluded. Its four plots and independent checks
pass, as does strict saved-output comparison. The campaign exporter adds only the general `filter_candidate_prepare`
phase to the existing explicit native-stage allowlist; this phase is exercised
by resident candidate preparation. Its hash is recorded separately in the manifest.

Repeat the rendering command for `filter-only` and `combined`. To regenerate the
combined residual-gap data, run `python <CAMPAIGN>/tools/analyze_gaps.py --input <CAMPAIGN>/profiles/combined/live-intervals.json.gz --output <CAMPAIGN>/metrics/combined-profile-gaps.json`.
These gaps are missing selected application kernel/copy activity, not proof that
the whole device is idle. Host-range/monitor overlap does not establish a cause.

[Clock checks](clock-verification.json) independently confirm that both session-origin
metadata fields agree and the six NVTX marker timestamps fall inside matching
wrapper before/return envelopes. GPU timestamps stay unchanged. Native host and
monitor timestamps subtract the session origin; marker emission latency is not
a clock correction. Exact counts/unions describe that fixed mapping. Host/GPU
phase membership has separate boundary uncertainty: the combined PSD total is
4,171 with a conservative 4,149–4,173 envelope range, while the old-loop and
ordered-Pallas counts are invariant; see the [kernel audit](../metrics/psd-kernel-verification.json).
