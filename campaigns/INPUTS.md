# Benchmark inputs

The complete input contract is [inputs.json](inputs.json). File names are local
to the directory you choose; no bank or frame data are stored in this repository.

The frames are the **cleaned GW170817-v2 release**, 2048 seconds starting at GPS
1187007040, sampled at 4096 Hz. The benchmark commands resample to 2048 Hz.
Use the exact H1/L1 URLs in the manifest, rather than the latest GW170817 release:
the event API distinguishes cleaned data from data containing the original L1
glitch. [GWOSC release](https://gwosc.org/eventapi/html/O1_O2-Preliminary/GW170817/v2/),
[file API](https://gwosc.org/api/v2/event-versions/GW170817-v2/strain-files?format=api).

GWOSC distributes its data under CC BY 4.0; follow its
[attribution guidance](https://gwosc.org/acknowledgement/), including the cited
open-data paper and dataset DOI [10.7935/K5B8566F](https://doi.org/10.7935/K5B8566F).
The O2 bank is fetched from the pinned `gwastro/pycbc-config` commit used by the
[original PyCBC Live example](https://github.com/gwastro/pycbc/blob/master/examples/live/run.sh).
An explicit redistribution license for that bank has not been established here,
so the XML and derived banks are downloaded/generated locally and are not
rehosted as release assets.

## Fetch and verify public files

From the repository root:

```sh
python tools/prepare_inputs.py plan
python tools/prepare_inputs.py fetch --input-dir inputs
python tools/prepare_inputs.py verify --input-dir inputs --scope public
```

`plan` performs no downloads or generation. `fetch` downloads only the XML and
two GWF files, verifies SHA-256 and size before accepting them, and checks existing
files without replacing them. Any mismatch fails. All three public URLs were
streamed and verified against the manifest on 2026-10-09.

## Generate the two benchmark banks

Input generation uses the historical original PyCBC checkout at commit
`40e94792b3edf59f39b18b65102b28a4f74433a7`, separate from the source revisions being
benchmarked. Use a clean checkout with its runtime already installed and built.
The pinned generation environment is Python 3.11.9, NumPy 1.26.4, h5py 3.11.0,
SciPy 1.13.0, LALSuite 7.21 and igwn-ligolw 2.1.1. This is the **input-generation**
environment; application benchmark dependencies are recorded with each campaign.

Obtain that source and select your Python 3.11.9 interpreter. Install the source
and pinned packages in an isolated environment; PyCBC's
[installation guide](https://pycbc.org/pycbc/latest/html/install.html) covers
system build dependencies.

```sh
git clone https://github.com/gwastro/pycbc.git original-pycbc
git -C original-pycbc checkout --detach 40e94792b3edf59f39b18b65102b28a4f74433a7
export INPUT_SOURCE="$PWD/original-pycbc"
export INPUT_PYTHON=/path/to/generation-environment/bin/python
"$INPUT_PYTHON" -m pip install -e "$INPUT_SOURCE" \
    numpy==1.26.4 h5py==3.11.0 scipy==1.13.0 lalsuite==7.21 igwn-ligolw==2.1.1
python tools/prepare_inputs.py plan --input-dir inputs \
    --input-source "$INPUT_SOURCE" --python "$INPUT_PYTHON"
python tools/prepare_inputs.py generate --input-dir inputs \
    --input-source "$INPUT_SOURCE" --python "$INPUT_PYTHON"
python tools/prepare_inputs.py verify --input-dir inputs
```

`generate` requires both options explicitly and runs selection, conversion and
compression in that interpreter. It refuses a different revision, dirty checkout,
different recorded dependency versions, or existing generated outputs. It sets
the recorded single-thread environment and `PYTHONHASHSEED=0`.

The recipe is:

1. Convert the pinned XML using `bin/bank/pycbc_coinc_bank2hdf`.
2. Keep rows with `mass1 <= 10`, `mass2 <= 3`, chirp mass `>= 1.2`, and both
   aligned spin magnitudes `<= 0.2`. These cuts yield 32655 rows. Permute with
   `numpy.random.default_rng(20260919)`, retain the first 1536 distinct
   `(mass1, mass2, spin1z, spin2z)` tuples, then sort their source row IDs.
3. Write those rows as `o2-subset-1536.hdf`, preserving root attributes. Copy its
   **first 1024 sorted rows** to `o2-subset-1024.hdf` for Live. Selecting 1024 rows
   independently from the random permutation would produce a different bank.
4. Compress the 1536-row bank with `bin/pycbc_compress_bank` using IMRPhenomD,
   `aLIGOZeroDetHighPower`, sample rate 2048, segment length 512, low-frequency
   cutoff 30, single precision, `inline_linear`, `mchirp`, one process and
   tolerance 0.001. `plan` prints the exact CLI arguments.
5. Check all three generated SHA-256 values and sizes before writing a
   `generation.json` receipt with source row IDs and dependency versions.

HDF5 bytes can depend on serialization and dependency versions. A mismatch is
reported as a different input; the script never rewrites expected hashes to
accept it. Preserve any failed output for diagnosis or use a new output directory
for a subsequent attempt.

The retained 2026-10-08 receipt records an exact regeneration of the 1536-row
subset and compressed bank. The Live bank's six datasets, dtypes and attributes
match the first 1024 rows of that subset; an independent in-memory recreation on
2026-10-09 also matched its full file hash. Those checks establish the recovered
recipe's provenance. A new full compression run using this CLI has not been
performed as part of publication preparation.
