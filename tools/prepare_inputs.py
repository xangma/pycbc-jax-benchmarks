#!/usr/bin/env python3
"""Fetch, generate, and verify the pinned benchmark inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen


MANIFEST = Path(__file__).resolve().parents[1] / "campaigns" / "inputs.json"
FIELDS = ("mass1", "mass2", "spin1z", "spin2z")
THREAD_ENV = {
    "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(directory, asset):
    path = directory / asset["filename"]
    if not path.is_file():
        raise ValueError(f"Missing input: {asset['filename']}")
    actual = sha256(path)
    if actual != asset["sha256"] or path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"Input mismatch: {asset['filename']}; SHA-256={actual}")
    return {"filename": asset["filename"], "sha256": actual, "verified": True}


def fetch(directory, assets):
    directory.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        path = directory / asset["filename"]
        if path.exists():
            verify_file(directory, asset)
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as output, urlopen(asset["url"], timeout=60) as stream:
                shutil.copyfileobj(stream, output)
            if (sha256(temporary) != asset["sha256"]
                    or Path(temporary).stat().st_size != asset["size_bytes"]):
                raise ValueError(f"Download mismatch: {asset['filename']}")
            if path.exists():
                raise FileExistsError(f"Input appeared during download: {asset['filename']}")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return [verify_file(directory, asset) for asset in assets]


def select_rows(source, count=1536, seed=20260919):
    """Select unique physical templates before sorting their source row IDs."""
    import numpy as np

    arrays = [source[field][:] for field in FIELDS]
    mass1, mass2, spin1z, spin2z = arrays
    chirp = (mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
    eligible = np.flatnonzero(
        (mass1 <= 10) & (mass2 <= 3) & (chirp >= 1.2)
        & (np.abs(spin1z) <= 0.2) & (np.abs(spin2z) <= 0.2)
    )
    chosen, seen = [], set()
    for index in np.random.default_rng(seed).permutation(eligible):
        key = tuple(values[index].item() for values in arrays)
        if key not in seen:
            chosen.append(int(index))
            seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Need {count} distinct templates; found {len(chosen)}")
    return sorted(chosen), len(eligible)


def copy_rows(source, output, rows):
    import h5py

    for name, value in source.items():
        if isinstance(value, h5py.Dataset):
            output.create_dataset(name, data=value[rows])
        else:
            source.copy(name, output)
    for key, value in source.attrs.items():
        output.attrs[key] = value


def validate_source(source, revision):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(source), *arguments], text=True,
        ).strip()

    if git("rev-parse", "HEAD") != revision:
        raise ValueError(f"Input generation requires source revision {revision}")
    if git("status", "--porcelain"):
        raise ValueError("Input generation requires a clean source checkout")
    for relative in ("bin/bank/pycbc_coinc_bank2hdf", "bin/pycbc_compress_bank"):
        if not (source / relative).is_file():
            raise ValueError(f"Input source is missing {relative}")


def commands(python, source, directory, manifest):
    xml = directory / manifest["public_assets"][0]["filename"]
    conversion = [
        python, str(source / "bin/bank/pycbc_coinc_bank2hdf"),
        "--bank-file", str(xml), "--output-file", str(directory / ".o2-full.hdf"),
    ]
    compression = [
        python, str(source / "bin/pycbc_compress_bank"),
        "--bank-file", str(directory / "o2-subset-1536.hdf"),
        "--output", str(directory / "o2-compressed-1536.hdf"),
        *manifest["generation"]["compression_arguments"],
    ]
    return conversion, compression


def generate(directory, source, manifest):
    import h5py

    generation = manifest["generation"]
    validate_source(source, generation["source_revision"])
    runtime = {name: importlib.metadata.version(name) for name in generation["dependencies"]}
    if runtime != generation["dependencies"] or platform.python_version() != generation["python_version"]:
        raise ValueError("Generation runtime differs from campaigns/inputs.json; use the pinned environment")
    for asset in manifest["public_assets"]:
        verify_file(directory, asset)
    for asset in manifest["generated_assets"]:
        if (directory / asset["filename"]).exists():
            raise FileExistsError(f"Refusing to overwrite {asset['filename']}")
    if (directory / "generation.json").exists():
        raise FileExistsError("Refusing to overwrite generation.json")
    conversion, compression = commands(sys.executable, source, directory, manifest)
    # Keep incomplete conversion output inside a private temporary directory.
    with tempfile.TemporaryDirectory(prefix=".conversion-", dir=directory) as temporary:
        full = Path(temporary) / "full.hdf"
        conversion[-1] = str(full)
        subprocess.run(conversion, check=True, cwd=source)
        validate_source(source, generation["source_revision"])
        with h5py.File(full, "r") as bank:
            rows, eligible = select_rows(bank)
            if eligible != generation["selection"]["eligible_rows"]:
                raise ValueError("Eligible template count differs from the recorded input recipe")
            with h5py.File(directory / "o2-subset-1536.hdf", "x") as subset:
                copy_rows(bank, subset, rows)
    verify_file(directory, manifest["generated_assets"][0])
    with h5py.File(directory / "o2-subset-1536.hdf", "r") as subset:
        with h5py.File(directory / "o2-subset-1024.hdf", "x") as live:
            copy_rows(subset, live, slice(0, 1024))
    verify_file(directory, manifest["generated_assets"][2])
    subprocess.run(compression, check=True, cwd=source)
    validate_source(source, generation["source_revision"])
    verified = [verify_file(directory, asset) for asset in manifest["generated_assets"]]
    receipt = {
        "source_revision": generation["source_revision"], "python_version": platform.python_version(),
        "dependencies": runtime, "source_rows": rows,
        "live_rows_of_sorted_1536_subset": [0, 1024], "assets": verified,
    }
    with (directory / "generation.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "fetch", "verify", "generate"))
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"))
    parser.add_argument("--scope", choices=("public", "generated", "all"), default="all",
                        help="Which assets verify checks (default: all)")
    parser.add_argument("--input-source", type=Path, help="Clean original PyCBC checkout for bank generation")
    parser.add_argument("--python", help="Explicit Python executable in the pinned generation environment")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    directory = args.input_dir.expanduser().resolve()
    if args.mode == "plan":
        conversion, compression = commands(
            args.python or "<python>", args.input_source or Path("<input-source>"),
            args.input_dir, manifest,
        )
        result = {"downloads": manifest["public_assets"], "conversion": conversion,
                  "selection": manifest["generation"]["selection"], "compression": compression,
                  "live_subset": "Copy the first 1024 rows of the sorted 1536-row subset",
                  "expected_outputs": manifest["generated_assets"]}
    elif args.mode == "fetch":
        result = fetch(directory, manifest["public_assets"])
    elif args.mode == "verify":
        groups = ("public_assets", "generated_assets") if args.scope == "all" else (args.scope + "_assets",)
        result = [verify_file(directory, asset) for group in groups for asset in manifest[group]]
    else:
        if args.input_source is None or args.python is None:
            parser.error("generate requires explicit --input-source and --python")
        source = args.input_source.expanduser().resolve()
        validate_source(source, manifest["generation"]["source_revision"])
        if not args.worker:
            environment = {**os.environ, **THREAD_ENV, "PYTHONPATH": str(source)}
            command = [args.python, str(Path(__file__).resolve()), "generate", "--worker",
                       "--input-dir", str(directory), "--input-source", str(source), "--python", args.python]
            return subprocess.run(command, env=environment).returncode
        result = generate(directory, source, manifest)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Input preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
