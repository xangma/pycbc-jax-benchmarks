# PyCBC JAX benchmarks

Reproducible workloads and dated evidence for the PyCBC JAX stack. General usage,
reference switches and the benchmark protocol belong in the
[PyCBC documentation](https://github.com/xangma/pycbc/blob/jax-pr47-performance-evidence/docs/jax.rst).
Measurements, plots and campaign receipts live here so they can be cited from PRs
without adding machine-specific results to the PyCBC source stack.

The [Live GPU-gap diagnostic](reports/2026-10-09-live-gaps/README.md) compares
the frozen JAX baseline with three focused changes. Median process wall time for
the same LAL-waveform replay falls from 66.90 s to 52.31 s across three samples;
all saved scientific outputs match exactly. Fresh Nsight timelines explain the
removed work and remaining gaps. This is a diagnostic comparison with explicit
validation limits. Its [dated release](https://github.com/xangma/pycbc-jax-benchmarks/releases/tag/2026-10-09-live-gaps)
preserves the protocol, inputs, observers and portable plotting data separately
from the earlier full-stack campaign.

The [9 October 2026 report](reports/2026-10-09/README.md) contains fresh Inspiral
and Live measurements, seven Nsight timeline views, portable plotting data and
explicit validation limits. The report measures the complete stack at
`366c7a955570deabaa1219d353f7847b21384698`; it does not measure each branch's
incremental performance. It establishes **no qualified CPU/GPU speedup or
steady-state capacity claim**. The default full-bank outputs still differ from
pristine PyCBC; a two-template all-original-route control matches exactly within
its saved-output scope.

| Evidence | Stable reference |
| --- | --- |
| Readable report and figures | [Benchmark site](https://xangma.github.io/pycbc-jax-benchmarks/reports/2026-10-09/) |
| Frozen report, protocol, inputs, code and plotting data | [Dated release](https://github.com/xangma/pycbc-jax-benchmarks/releases/tag/2026-10-09-full-stack) |
| General four-arm method | [Canonical PyCBC protocol](https://github.com/xangma/pycbc/blob/jax-pr47-performance-evidence/docs/jax_benchmark_protocol.rst) |
| Exact input acquisition and bank generation | [Input recipe](campaigns/INPUTS.md) |
| Timing boundaries used in the dated report | [Archived observer controllers](campaigns/2026-10-09-observers/README.md) |
| Regenerate timeline plots without Nsight | [Profile data and renderer](reports/2026-10-09/profiles/README.md) |

## Reproduce a workload

Use an isolated application environment outside the source checkouts. PyCBC
requires its normal build dependencies, including LALSuite. CPU and CUDA
application environments may be separate; select their interpreters explicitly.
The dated [runtime manifest](reports/2026-10-09/provenance.json) records observed
versions and extension hashes. A new build records its own hashes rather than
claiming byte identity with the measured runtime.

```sh
git clone https://github.com/xangma/pycbc.git sources/candidate
git -C sources/candidate checkout --detach jax-benchmark-2026-10-09
git clone https://github.com/xangma/pycbc.git sources/original
git -C sources/original checkout --detach jax-benchmark-original-2026-10-09
python tools/prepare_inputs.py plan
```

Follow the [input recipe](campaigns/INPUTS.md) to fetch and verify the public
frames and bank XML and regenerate the exact bank files. The input-generation
checkout and environment are separately pinned. Install/build each benchmark
source in its matching application environment. Keep the source checkout clean;
ignored native extensions are hashed before and after execution.

The native-command runner uses only the Python standard library. It launches
one excluded prime per arm, then at least three fresh processes in rotated arm
order, followed by a separate profile when configured. Each arm gets private
caches and temporary directories. Scientific/backend selectors come from the
manifest; inherited PyCBC/JAX settings are cleared. Source imports, exact input
hashes, return codes, deadlines and descendant cleanup are checked. Resolved
receipts remain private because they contain paths, commands and process IDs.

```sh
python -m pycbc_jax_benchmarks run campaigns/inspiral-1536-four-arm.json \
    --sources sources --inputs inputs --output results/four-arm \
    --python original-cpu="$CPU_PYTHON" --python branch-cpu="$CPU_PYTHON" \
    --python jax-cpu="$JAX_CPU_PYTHON" --python cuda="$CUDA_PYTHON" --plan
```

Set those interpreter variables to your installed runtimes. Add `--cpus ARM=IDs`
for each arm to choose physical-core-aware affinity for your hardware. Inspect
`--plan`, then omit it to execute into a new output directory. The plan checks
source/input hashes and does not import or launch the applications. Diagnostic
manifests for [Inspiral](campaigns/inspiral-1536-diagnostic.json) and
[Live](campaigns/live-1024-diagnostic.json) define the corresponding smaller
measurement campaigns. Their `work.expected` values are contracts to check,
not observed completion evidence.

The general runner measures native CLI launch through exit and log closure.
It excludes source/input guards and preflight imports. **It does not reproduce
the dated observer timings or named profile windows**, which include passive
observers, in-process checks, final synchronization and Live suffix accounting.
Use the archived controllers for those boundaries. Profiling and unprofiled
samples are always separate.

Successful execution leaves scientific qualification, completed warm-up, verified
work and realtime capacity marked unavailable. Validate them using the protocol
before making a performance claim. The four-arm manifest is a reproducible
starting point, not evidence that those four arms have already passed.

## Publish a new campaign

Freeze a new source revision, input hashes, manifest and protocol version before
running. Keep failed runs and unmet gates explicit. Publish a new dated directory
and release; do not overwrite an earlier campaign. A PR should cite the release,
name the exact relevant workload and timing scope, and state qualification limits.
A full-stack result must not be described as an individual PR's measured speedup.

Raw Nsight/SQLite captures and bank/frame data are not distributed here. The
portable profile data preserve exact selected application intervals and their
independent checks without private metadata. See [input attribution](campaigns/INPUTS.md).
Source code is distributed under [GPL version 3](LICENSE), retaining the license
and notices of PyCBC-derived files.

To check the runner and rebuild the static site:

```sh
python -m unittest -v test_runner
python -m pip install Markdown==3.10.3
python tools/build_site.py
```
