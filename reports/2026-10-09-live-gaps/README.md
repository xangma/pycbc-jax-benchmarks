# Live GPU-gap diagnostic — 2026-10-09

A fixed 1,024-template, two-detector replay compares the frozen JAX baseline,
a filter-admission correction and a combined JAX optimization. Each arm uses
two MPI ranks, one RTX 4090, LAL TaylorF2 waveforms and default JAX conditioning.

| Arm | Process wall, median [min–max], s | Post-two-valid-block suffix, median [min–max], s | Exact saved-output gate |
| --- | ---: | ---: | --- |
| Baseline | 66.9047 [66.8552–66.9537] | 30.7547 [30.7100–30.8247] | Pass |
| Filter-only | 66.4793 [66.2338–66.7902] | 30.7411 [30.7209–30.7424] | Pass |
| Combined | 52.3076 [51.2541–52.3116] | 16.7178 [16.0256–16.7285] | Pass |

The combined median whole-process wall is **21.82% lower** than baseline.
The suffix is descriptive: the shared data epoch can include earlier-block
backlog on the other rank, so its retained stage work differs across arms.

Each measured arm has three fresh MPI jobs after one excluded prime; profiling
is separate. Arms run in groups, not rotated/interleaved order. These are
**diagnostic comparisons**, with tracing/loading still present in the baseline
and filter-only suffixes; rank 0 still traces/lowers/loads in the combined
suffix. They establish no compile-free capacity, pristine-CPU
speedup or complete science qualification. The filter-only median wall difference
is 0.425 s; the suffix difference is only 0.014 s.

Measured mechanisms:

- **PSD launches:** every baseline PSD refresh on each rank has 7,954
  `loop_add_fusion_1` events (`2 × 3977`) and 7,956 `loop_add_fusion` events
  (`2 × 3978`). Across 26 refreshes per rank these families account for 827,320
  events, 90.31% of all captured kernels, but only 0.820 s summed GPU duration.
  The launch signature associates them with the ordered scan; it does not by
  itself establish wall-time cost or explain every gap.
- **Coincidence specialization:** baseline rank 0 gains 44 foreground-core JIT
  entries over 23 cluster calls and 29 match-group entries over 33 query calls in
  each suffix. Foreground-core trace/lower/backend-cache durations have medians
  5.022/2.675/1.699 s. Persistent-cache hits coexist with fresh-process tracing,
  lowering and executable loading; callback durations can nest and are not
  additive wall-time attribution.
- **Combined:** the 827,320-event PSD launch signature becomes 104 events
  (two `loop_add_fusion` per refresh; no `loop_add_fusion_1`). PSD-associated
  launches fall from 831,296 to 4,171; the combined boundary-envelope range is
  4,149–4,173. Ordered Pallas scan launches rise from zero to 104 (two per
  refresh). These counts associate kernels with host-stage time windows under
  the recorded clock mapping; boundary uncertainty is recorded in the
  [audit](metrics/psd-kernel-verification.json). The two named foreground-core/join
  functions have no named trace/lower/backend notifications in any timing suffix;
  their names are absent from cache-size snapshots. Other functions still trace,
  lower and load. Generic kernel names remain
  elsewhere, and this does not attribute all wall-time savings.
- **Resident filtering:** all three filter-only replays prepare the resident
  candidate/veto functions; baseline compatibility preparation disappears.
  Rank 1 has no trace/lower/backend-cache notifications in any filter-only suffix,
  while rank 0 still does. Exact saved outputs remain unchanged. See
  [mechanisms](mechanisms.json), [route evidence](metrics/filter-route-observations.json)
  [duration distributions](metrics/baseline-duration-cache.json) and the
  [independent kernel audit](metrics/psd-kernel-verification.json).

| Separate Nsight profile | CLI scope / recorded kernel+copy union, s | Suffix / recorded union, s | Evidence |
| --- | ---: | ---: | --- |
| Baseline | 81.435 / 2.774 | 41.748 / 2.272 | [Timeline](profiles/baseline/warm.png), [400 ms](profiles/baseline/detail.png), [data](profiles/baseline/live-intervals.json.gz) |
| Filter-only | 81.951 / 3.241 | 41.344 / 2.688 | [Timeline](profiles/filter-only/warm.png), [400 ms](profiles/filter-only/detail.png), [data](profiles/filter-only/live-intervals.json.gz) |
| Combined | 60.254 / 1.790 | 19.192 / 1.379 | [Timeline](profiles/combined/warm.png), [400 ms](profiles/combined/detail.png), [data](profiles/combined/live-intervals.json.gz) |

Recorded union excludes memsets and is not occupancy or whole-device utilization;
a larger union is not itself an improvement. Host ranges have no stage fences.
All three profiles exclude 132 unmatched stage starts. Exact counts, transfer bytes,
integer unions and ordered GPU-event hashes pass independent read-only SQLite
checks. [Regenerate all plots](profiles/README.md) from the portable data.

Baseline repeats/profile and both candidates’ prime/all samples/profile pass
strict saved-output comparison: 33 HDF files, 1,067 datasets, 203 attributes,
52 PSD datasets and 693 saved single-trigger rows per run. Hash seed 4 is fixed
identically, with no trigger/label sorting, scientific exclusions or relaxed
tolerances. Complete conditioned strain/internal decisions/retained state and a
pristine-CPU full Live comparison are outside this saved-output gate.
[Validation details](validation.json) retain those limits. Separately,
[521 focused regression cases passed](validation-tests.json), including 242 CUDA
parameter cases, with zero skips or failures, on the bound combined source.

[Source bindings](sources.json), [inputs](inputs.json), [all metrics](metrics.json)
and the [exact diagnostic procedure](PROTOCOL.md) make the campaign reproducible.
The same runtime reuses 11 prebuilt native extensions and generated version
metadata from the baseline build; candidates are not clean native rebuilds.
[Runtime hashes](runtime-overlay.json) and [software provenance](software.json)
are separate from source revision/tree
identity. The [canonical PyCBC protocol](https://github.com/xangma/pycbc/blob/jax-pr47-performance-evidence/docs/jax_benchmark_protocol.rst)
defines the broader qualification method; this campaign records its narrower
observer boundaries explicitly. Earlier benchmark releases remain separate.
