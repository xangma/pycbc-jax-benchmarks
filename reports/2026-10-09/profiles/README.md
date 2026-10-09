# Reproduce the profile figures

These data and figures describe PyCBC commit
`366c7a955570deabaa1219d353f7847b21384698`. They are instrumented diagnostics;
use the campaign's separate unprofiled samples for performance comparisons.

The two `*-intervals.json.gz` files contain integer event boundaries, application
or MPI rank labels, GPU device indices, copy directions/bytes and selected host
phase names. They contain no strain samples, arbitrary kernel labels, launch
commands, PIDs, hostnames, private paths or raw Nsight tables. Raw `.nsys-rep` and
SQLite captures are retained privately. Export input hashes connect the public
derived data to those captures without publishing them.

From the repository root, install Python and Matplotlib, then run:

```sh
python -m pip install matplotlib==3.11.1
python tools/render_profile.py \
  --input reports/2026-10-09/profiles/inspiral-intervals.json.gz \
  --output reproduced/inspiral
python tools/render_profile.py \
  --input reports/2026-10-09/profiles/live-intervals.json.gz \
  --output reproduced/live
```

The renderer checks every plotted scope against the original verified summary:
exact integer kernel, copy and combined interval unions; overlapping event
counts; and completed transfer bytes per application/rank, device and direction.
It aborts on disagreement. Matplotlib or font changes can change PNG bytes while
these numerical checks remain exact. The published figures used Matplotlib
3.11.1. `manifest.json` records the portable file checksums and verification.
`independent-verification.json` records a second implementation's direct SQLite
union/count checks, which also agree with the public renderer.

| View | Inspiral | Live |
| --- | --- | --- |
| `whole.png` | Selected application GPU/host extent | Selected application GPU/host extent |
| `search.png` | First through last of 24 filtering batches | Combined CLI entry through final completion scope |
| `warm.png` | Repeated batch 13/24, without proof of completed warm-up | Common valid-data suffix after two valid blocks; compilation/tracing continues |
| `detail.png` | — | Central 400 ms of the same suffix |

“Warm” is a legacy file/window key, **not a qualified warmed throughput claim**.
Times are relative to the capture's session start. GPU lanes show selected
application kernel and memcpy intervals, merging overlapping streams. Memsets
are excluded. Combined interval union measures recorded activity duration; it
is not GPU occupancy, saturation or whole-device utilization. Transfer bytes
count each whole copy when its completion lies in the window, including a copy
that starts before the window.

Host phases measure submission and waiting without inserted stage completion
fences. Only the explicitly marked final completion scope has a completion
fence. JAX callback points mark tracing, lowering, backend and cache events;
backend notifications can include cache retrieval and are not cold compilation
durations. Live has **132 native stage starts without matching ends**, which
are excluded rather than given invented durations.

For dense lanes, the plot merges events into at most 12,000 display bins per
lane. A displayed bin means that an event intersects the bin; it does not mean
the GPU was active throughout the bin. Exact event intervals remain available
in the compressed files and all reported unions use those exact intervals.
Compact host lanes show allowlisted native and observer phases rather than raw
compiler labels. This is a presentation filter, not causal GPU attribution.

## Export a new private capture

`tools/export_profile.py` requires a read-only Nsight SQLite capture, its
matching observer reports and a verified summary that has already established
window boundaries and any mixed NVTX clock domains. It rejects unknown native
stage names and monitoring events; review its explicit allowlists when updating
instrumentation. GPU events are admitted only for application PIDs identified
in the matching observer reports, then relabelled `search`, `rank 0`, etc.
It does not export Nsight's `StringIds` table or arbitrary metadata.

For example, with a private collection organized using these generic filenames:

```sh
python tools/export_profile.py --app inspiral \
  --revision 366c7a955570deabaa1219d353f7847b21384698 \
  --sqlite private-capture/inspiral.sqlite \
  --summary private-capture/inspiral-verified-summary.json \
  --report private-capture/inspiral-observation.json \
  --output inspiral-intervals.json.gz
python tools/export_profile.py --app live \
  --revision 366c7a955570deabaa1219d353f7847b21384698 \
  --sqlite private-capture/live.sqlite \
  --summary private-capture/live-verified-summary.json \
  --report private-capture/live-profile-status.json \
  --report private-capture/rank-0.json \
  --report private-capture/rank-1.json \
  --stages private-capture/rank-0.stages.jsonl \
  --stages private-capture/rank-1.stages.jsonl \
  --output live-intervals.json.gz
```

The exporter uses only the Python standard library. Run the renderer on each
new export before publication and inspect all public strings and images.
Publishing a new capture also requires new immutable source/input/runtime
manifests and independent boundary verification; changing a filename alone
does not establish a valid campaign.
