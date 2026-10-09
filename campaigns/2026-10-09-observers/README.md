# Archived passive-observer campaigns

These are the exact observer programs used for the 2026-10-09 fresh profiles,
with portable configuration. They invoke the dedicated PyCBC CLI once per
process and observe its existing operations. This reproduces the **observer
measurement scope**, which includes observer imports, source/input checks and
reporting. It is different from timing a native CLI subprocess directly with
this repository's generic runner.

The four Python programs and two source manifests are copied byte for byte.
[provenance.json](provenance.json) records their SHA-256 values, the original
resolved configuration hashes and the public template hashes. The retained
harness manifests are prelaunch freeze receipts; their status fields are not
final run statuses. The successful
Inspiral configuration uses batch size **64**; the earlier 256 setting failed
with a CUDA allocation error and is not substituted here. Live uses two MPI
ranks. No strain, template banks, native binaries, cache files or raw profiles
are bundled in this directory.

## Source and inputs

Use a clean, built PyCBC checkout at commit
`366c7a955570deabaa1219d353f7847b21384698`, with tree
`d43edf428d9bba23fc0243f41a4c66faf7d4de5f`. A different revision, a dirty checkout,
or different input bytes is a different campaign. The source manifests retain
the original tracked-source guards: Inspiral checks 1288 tracked files; Live's
per-process guard checks its original 623 Python/CLI files and records loaded
PyCBC module origins and hashes. Configuration additionally checks the full
1288-file manifest for both applications.

Prepare the files described in [the input recipe](../INPUTS.md) into one input
directory. The required basenames and hashes are in [inputs.json](inputs.json):

- `o2-compressed-1536.hdf`: 1536 distinct stored LAL IMRPhenomD templates.
- `o2-subset-1024.hdf`: 1024 distinct physical rows, generated with LAL TaylorF2
  by the Live CLI.
- `H-H1_LOSC_CLN_4_V1-1187007040-2048.gwf` and
  `L-L1_LOSC_CLN_4_V1-1187007040-2048.gwf`: the pinned cleaned GW170817-v2 frames.

Both applications resample to 2048 Hz and check the actual complex64 filtering
precision. `waveform` is the only selected original implementation; the other
JAX stages retain their device defaults. Inspiral uses `device_linear`
decompression and additionally runs a separate `inline_linear` diagnostic.
Both select NumPy for original FFT validation. The waveform-provider import
guards remain unchanged.

The application interpreter must already have its matching PyCBC extensions,
generated `pycbc/version.py`, GPU JAX, LALSuite, NumPy, h5py and other PyCBC
runtime dependencies. Live additionally requires MPI/mpi4py, the optional
`cuda-zlib` runtime used by the GWF decoder, and its CUDA compiler. Both need
Linux `taskset`, Nsight Systems and an NVTX shared library; Live needs `mpirun`
on `PATH`. Select two CPU IDs that are allowed by the system and correspond to
the intended physical cores. The controller requests affinity; topology and
actual process bindings still require independent observation.

[published-runtime-files.json](published-runtime-files.json) retains the native
file hashes from the published build. Configuration creates a **new** runtime
manifest from the selected interpreter's ABI extensions and generated version
file. It records the interpreter, installed distribution metadata and tool
hashes. A rebuilt extension is never labeled as the published binary. The recorded application runtime versions are in
[published-runtime-environment.json](published-runtime-environment.json):
Python 3.12.8, NumPy 2.2.6, JAX/jaxlib 0.11.2; the Inspiral receipt also records
LAL 7.7.1, LALSimulation 6.2.1 and LALFrame 3.0.8. This is an observed inventory,
not a complete dependency/build lock. Package metadata from configuration is
also an inventory, not a successful GPU/import preflight. Inspiral rechecks the new runtime manifest at each phase. Live retains its
original loaded-module checks; its new runtime manifest is configuration
evidence, rather than a new per-phase guard. The optional codec's generated
cache is private to the configured campaign.

## Configure without launching a search

From this repository's root, supply your local locations and bindings:

```sh
ARCHIVE="$PWD/campaigns/2026-10-09-observers"
python "$ARCHIVE/configure.py" \
    --source "$SOURCE" --inputs "$INPUTS" --output "$OUTPUT" \
    --python "$PYTHON" --cpus "$CPUS" --gpu "$GPU" \
    --nvtx-library "$NVTX_LIBRARY" --nsys "$NSYS" --nvcc "$NVCC"
```

`SOURCE` is the pinned built checkout, `INPUTS` holds the prepared files, and
`OUTPUT` is a **new private directory outside this repository and the
PyCBC checkout**. `PYTHON`
is the application interpreter. `CPUS` contains exactly two distinct CPU IDs,
separated by a comma; Inspiral uses the first, while Live's two ranks share
both. `GPU` selects one CUDA ordinal or UUID through `CUDA_VISIBLE_DEVICES`;
`jax:cuda:0` consequently addresses the one visible GPU. `NVTX_LIBRARY` names
the existing shared library. `NSYS` and `NVCC` are executable paths or names on
`PATH`; `--nvcc` may be omitted when `CUDACXX` or `nvcc` on `PATH` selects it.

This command reads source/input bytes and runs only a standard-library
interpreter/ABI metadata query. It imports no PyCBC/JAX/LAL modules and starts
no search, MPI job or profile. It refuses missing matching native extensions,
changed archived assets, wrong source/input hashes and an existing output
root. It writes resolved private configs, copied frozen wrappers, a runtime
manifest and `configuration.json`, containing the exact next commands.

The Live controller invokes the executable name `nsys`. Configuration supplies
a private `PATH` shim for the explicitly selected tool, preserving the
controller bytes. No other executable is overridden. Resolved configs and
receipts contain your local paths; keep them outside the public source tree.

## Run the archived controllers

Use the emitted commands, or the equivalent commands below. Each controller
must run on the configured Linux GPU system; configuring on its own is not an
application validation.

Inspiral plan and separate import/input preflight:

```sh
"$PYTHON" "$OUTPUT/inspiral/run_campaign.py" \
    --config "$OUTPUT/inspiral/campaign_config.json"
"$PYTHON" "$OUTPUT/inspiral/run_campaign.py" \
    --config "$OUTPUT/inspiral/campaign_preflight.json" \
    --execute --preflight-only
```

The first command only prints a plan. The second uses a separate preflight
output root, imports the GPU runtime and verifies the bank; it runs no search.
Then the unchanged controller performs preflight, one excluded prime, three
fresh-process samples, a separate inline-decompression diagnostic, and a
separate Nsight profile followed by export/statistics:

```sh
"$PYTHON" "$OUTPUT/inspiral/run_campaign.py" \
    --config "$OUTPUT/inspiral/campaign_config.json" --execute
```

Live's archived controller has `samples` and `profile` modes, with no standalone
plan/preflight mode. `configure.py` is its nonlaunching configuration check;
`--help` is available on both frozen programs. The samples mode runs one
excluded prime followed by three fresh two-rank jobs. The profile mode requires
that samples completed successfully:

```sh
"$PYTHON" "$OUTPUT/live/run-v2.py" \
    --config "$OUTPUT/live/config-v2.json" --mode samples
"$PYTHON" "$OUTPUT/live/run-v2.py" \
    --config "$OUTPUT/live/config-v2.json" --mode profile
```

All launches keep process IDs, commands, logs and stop commands. Do not run the
two applications concurrently on the same resources. Outputs are not reused;
retain failures and configure a new directory for another attempt. Controllers
and workers use private temporary/cache/output directories, except installed
runtime components and their normal process environment. No cache from the
published run is assumed.

## Timing and qualification

Inspiral's three sample full-process walls run from child launch to exit. They
include passive-observer imports, setup, cache activity, search and HDF output.
The recorded `run_time * (1 - setup_time_fraction)` is the existing application
host timer: it excludes setup/HDF writing and adds no device-completion fence.
It is not a synchronized GPU compute timer. The separate Nsight run is excluded
from the sample median. Its host ranges do not introduce per-stage GPU fences.

Live's full-process walls include MPI launch through worker exit, including
source/input checks and reporting. The separate native-CLI interval runs from
`runpy` entry through one final JAX completion boundary and MPI barrier. Its
suffix begins after the first **two valid analyzed blocks**, not after two full
replay warmups. Those first blocks are excluded from both suffix time and unique
detector work. Unique detector seconds come from interval unions on rank zero;
they are never counted again for the second rank. Host stage markers and one
NVTX clock marker per rank do not add per-stage device fences. Profile/controller
wall includes profiling/export overhead and is not a timing sample.

A prime and a fixed block cutoff do not establish fully warm execution. Live's
unchanged listener records JAX trace/lower/backend-duration and cache
notifications; backend-duration notifications can include persistent-cache
loads. It does not retain each duration or compilation key, and notifications
are not a complete census of cold compilation. The recorded suffix flag must be
reported with this limitation. The published suffixes contained such activity;
they were not qualified as fully warm throughput.

These controllers reproduce candidate diagnostics, not the full four-arm
benchmark protocol or its scientific gates. They do not supply pristine CPU,
branch CPU and JAX CPU comparisons, complete intermediate/selection evidence,
rotation across four arms, convergence, or full-machine capacity validation.
Repeated candidate outputs and a successful profile are useful checks but do
not establish those missing gates. Preserve exact repeat-comparison failures,
including output-order differences. The generic runner remains available for
new campaigns; its direct native-CLI full-wall measurement must not be relabeled
as this archived observer scope.

The configuration helper can be checked without PyCBC or a GPU:

```sh
python -m unittest discover -s campaigns/2026-10-09-observers -p test_configure.py
```

These checks exercise archive hashes, exact-source rejection, token expansion,
interpreter symlink handling and runtime-manifest construction with small
fixtures. They do not validate an actual application installation.
