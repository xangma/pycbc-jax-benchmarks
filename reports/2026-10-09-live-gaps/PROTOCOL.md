# Reproduce the diagnostic campaign

Use the source commits and production manifests in [sources.json](sources.json)
with the same runtime across arms. All candidates reuse the frozen baseline
build's 11 native extensions plus generated `pycbc/version.py` metadata, as
[hashed separately](runtime-overlay.json); they are not clean native rebuilds.
Generated version text does not identify a candidate source commit. The explicit
revision/tree and production hashes establish source identity.

The measured system has a Threadripper PRO
3995WX, 128 logical CPUs, 251 GiB RAM and one RTX 4090 (24,564 MiB). Allocate the
same two fixed CPU cores and one numerical thread per MPI rank. Other system
services remain running. The observed runtime versions are Python 3.12.8,
JAX 0.11.2 and NumPy 2.2.6. The [inherited observed baseline inventory](software.json)
also records jaxlib 0.11.2, SciPy 1.16.3, LALSuite 7.26.15, h5py 3.16.0, mpi4py
4.0.0 and Nsight Systems 2026.1.3.425. Only Python/JAX/NumPy are re-recorded by
the new rank receipts. These observations are not a complete environment lock;
record recreated environment and native binary hashes before rerunning validation.

Prepare the exact uncompressed Live bank and GW170817 frames using the benchmark
repository's `tools/prepare_inputs.py` and the pinned provenance/selection in
[inputs.json](inputs.json). Check all SHA-256 values before running. Frame input
is 4096 Hz; Live analyzes at 2048 Hz. The historical input-generation source
revision is separate from the measured JAX baseline.

For each arm, copy `observer/<arm>/config-template.json` to `config-v2.json` in
the same directory. Replace every angle-bracket token with your own resource
location or two distinct CPU IDs. `<SOURCE>` is that arm's frozen source checkout;
`<PYTHON>` is the common Python executable; `<INPUTS>` holds the verified files;
`<RESULTS>` is a new output directory; `<CACHE>` is a private cache directory unique
to the arm; `<CUDA>` and `<NVTX_LIBRARY>` identify the installed CUDA toolchain and
NVTX library. Keep the adjacent source manifest unchanged. The driver verifies
all recorded production blobs and input hashes before and after execution.
The scientific command and environment are fully listed in
[protocol.json](protocol.json).

From the benchmark repository root, with `<CAMPAIGN>` replaced by this campaign's
relative directory, run:

```sh
python <CAMPAIGN>/observer/run-v2.py --config <CAMPAIGN>/observer/baseline/config-v2.json --mode samples
python <CAMPAIGN>/observer/run-v2.py --config <CAMPAIGN>/observer/baseline/config-v2.json --mode profile
```

Repeat with the filter-only and combined configs, then apply the exact-output
validation gate before interpreting the comparison. All three arms have completed
prime, three timing samples and a separate profile. Existing result directories are never reused. The observer and
controller are the exact recorded scripts; their hashes are in `sources.json`.
Generated rank receipts and raw Nsight reports can contain local paths/PIDs and
must remain private. Publish only the explicitly sanitized derived evidence.

The samples command runs one excluded prime and three fresh MPI process pairs.
It retains that arm's persistent JAX/codec caches. The profile command requires
successful samples, runs a separate fresh pair under Nsight Systems, and exports
SQLite privately. The recorded command traces CUDA, NVTX and OS runtime, includes
CUDA graph nodes, and disables CPU sampling/context-switch capture. There are no
per-stage device fences. The wrapper performs one final JAX completion/effects
barrier and MPI barrier.

The full replay has 33 native blocks, of which 26 valid 8 s blocks per detector
cover 208 unique seconds each: 425,984 template-detector-seconds. The suffix drops
the first two valid blocks and covers 192 s per detector: 393,216
template-detector-seconds. Its wall boundary is the earliest rank reaching the
same data epoch through latest completion. Do not use suffix work with the full
launch-to-exit wall. The original replay's observed valid endpoint exceeds the
requested CLI endpoint and is retained identically in every arm.

The prime populates the persistent cache; it does not warm a new process's tracing,
lowering or executable-loading state. Check the observed callbacks and cache
counts inside each measured suffix before claiming a compile-free interval.
Callbacks' own measured overhead does not remove all logging, formatting,
scheduling or instrumentation cost. Profile timing is excluded from sample
summaries. Actual arm sampling is grouped, not interleaved.

Compare every nonempty saved HDF inventory, ordered dataset dtype/shape/value bits
and all attributes against the fresh fixed-seed baseline. Normalize only the
verified executable and output-location command fields. Do not sort triggers or
detector labels, drop scientific datasets or relax tolerances. Report the strict
saved-output gate separately from tests of internal retained state and any
complete pristine-CPU qualification. Keep failures with their exact differences.

To reproduce duration/cache aggregates from private rank receipts, run the supplied
`tools/analyze_duration_observers.py --input <RESULTS> --output <PRIVATE_ANALYSIS>`,
then `tools/aggregate_duration_samples.py --analysis <PRIVATE_ANALYSIS>/duration-cache-analysis.json --controller <RESULTS>/samples-status.json --output <AGGREGATES>`.
The detailed intermediate contains process/thread identities and stays private;
only the aggregate three-sample summary is portable. Callback durations can nest
and must not be summed as independent wall-time costs.
