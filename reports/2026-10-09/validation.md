# Validation — 9 October 2026

Fresh profiles preserve the saved scientific outputs of their unprofiled
reference runs. A separate two-template control also reproduces native original
PyCBC exactly with original calculation routes selected. **The default complete
searches are not scientifically qualified**, and these checks support no
qualified speedup, capacity or compilation-free steady-state claim.

The measured JAX source is `366c7a955570deabaa1219d353f7847b21384698`; the pristine
CPU source is `454ee900e6faca88a066e71ade489fca9fb221a5`. Waveforms use the original
PyCBC/LAL route; diffGW generation is disabled. Default conditioning remains on
the selected JAX device. See [validation.json](validation.json) for full-precision
differences, input/source identifiers, comparison outcomes and evidence hashes.

## Exact comparison scope

Comparisons preserve dataset order, dtype, shape and floating-point bits, and
check scientific attribute inventories and values. Compression and chunk layout
are outside this scientific-value check. Whole-file hashes can therefore differ
when excluded timing fields differ.

Inspiral excludes exactly four timing/rate datasets: `H1/search/run_time`,
`filter_rate_per_core`, `setup_time_fraction` and `templates_per_core` within the
same search group. Search start/end times and gating values remain checked.
Live excludes no datasets: the expected executable is verified before normalizing
its name and the output destination in `command_line`; every other argument and
attribute remains checked. No extra scientific exclusions or fitted tolerances
were introduced. Profiled timings are excluded from benchmark samples.

| Comparison | Saved output result |
| --- | --- |
| Inspiral sample 1 versus samples 2 and 3 | Exact: 18 scientific datasets each |
| Inspiral sample 1 versus separate Nsight profile | Exact: 18 scientific datasets |
| Live sample 1 versus sample 2 | Exact: 33 files, 1,067 datasets, 203 attributes |
| Live sample 1 versus separate Nsight profile | Exact: 33 files, 1,067 datasets, 203 attributes |
| Live sample 1 versus sample 3 | Strict failure: one detector-order string; all remaining datasets and attributes exact |

## Default Inspiral differences

The large workload completes 1,536 distinct compressed IMRPhenomD templates over
1,904 valid H1 seconds: five 512-second FFT segments at 2,048 Hz, with observed
complex64 signal arrays and batch size 64. The default device-decompression
search produces 3,674 triggers in each sample and the profile.

Changing only the decompression route to original `inline_linear`, while keeping
other numerical stages on JAX, produces 3,675 triggers. The fresh pristine CPU
search uses NumPy FFTs and original inline decompression and produces 3,676.
Matching template hash and sample-grid end time finds all 3,674 default-JAX
triggers in each comparison, with no duplicate or off-grid identities:

| Reference versus default JAX | Matched | Reference-only | JAX-only |
| --- | ---: | ---: | ---: |
| Current JAX with original inline decompression | 3,674 | 1 | 0 |
| Pristine original CPU | 3,674 | 2 | 0 |

| Matched saved field | Inline versus device decompression | Pristine CPU versus default JAX |
| --- | ---: | ---: |
| Maximum relative chi-square difference | 0.4284750453% | 0.9192256080% |
| Maximum absolute chi-square difference | 0.166595458984375 | 0.2530632019042969 |
| Maximum relative SNR difference | 0.008217674890% | 0.008184867116% |
| Maximum relative normalization, `sigmasq`, difference | 0.00001189873604% | 0.00001963275942% |
| Maximum circular coalescence-phase difference | 0.00007915496826171875 radians | 0.00007915496826171875 radians |

These are descriptive errors in the composed search, without renormalizing saved
chi-square values or evaluating a tolerance gate. The pristine comparison does
not isolate individual numerical stages. Per-field maxima can occur at different
triggers; subtracting maxima does not measure a stage's contribution. The two
missing default trigger identities already prevent complete output agreement.

## Bounded original-route agreement

The first two physically distinct bank rows retain IDs 0 and 1, their exact
parameters, header and compressed waveform arrays. Both native original CPU and
current JAX with all 298 registered original selectors selected complete the
same five segments and 1,904 valid H1 seconds and save five triggers. Their
18 scientific datasets match exactly; both attribute inventories are empty.
All 25 source/input/argument/workload contract checks pass. Selecting every
registered control does not imply this search exercises every registered API.

The current CLI's `device_linear` option remains present, but the selected
`decompress` reference control overrides that calculation with its original
route. Original CPU and JAX execution/batch/reference options differ explicitly;
their science arguments otherwise agree. The JAX signal dtype is observed as
complex64; native CPU intermediate signal dtype is not independently captured.

This proves composed original-route agreement for these **two templates and saved
outputs only**. The full 1,536-template all-original attempt was stopped with its
first 64-template batch still at segment 2 of 5 after approximately 11 minutes.
It remains incomplete and supplies no full-bank agreement result.

## Live ordering and timing limits

Live uses 1,024 distinct TaylorF2 templates, two MPI ranks and 2,048 Hz sampling.
The worker reports complex64 arrays. Its 33 blocks include 26 valid eight-second
blocks for each detector: 208 valid H1 seconds and 208 valid L1 seconds. Source,
input, argument and workload contracts pass for all samples and the profile.

The sample-3 strict failure is `foreground/type`: `H1-L1` becomes `L1-H1` in one
foreground file. All 1,014 numeric or structured datasets, the other 52 string
datasets and all compared attributes remain exact. Detector membership is the
same. Source tracing shows the native event manager constructs a detector set,
whose iteration order passes through combinations into the original/JAX label
join. Isolated hash-seed controls reproduce both labels, and the pristine native
source shares this construction. The failure remains recorded; no dataset was
excluded and no production fix was applied.

All recorded Live `zero_warm_compilation` flags are false. Compilation-free
steady-state timing is therefore uncertified; those flags alone do not prove
uncached compilation. No fresh pristine-CPU complete-search comparison is
available for this Live workload.

Full conditioned-strain arrays, complete PSD comparison coverage and early/later
decision audits are missing. Full-bank composed reference agreement is also
unestablished. Scientific qualification and qualified capacity/speedup claims
remain unavailable; diagnostic elapsed measurements do not relax these gates.
