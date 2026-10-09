"""Run pinned commands in fresh processes and preserve private receipts."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time


class CampaignError(RuntimeError):
    """Invalid campaign, changed evidence, or failed native command."""


UNAVAILABLE = {
    "scientifically_qualified": None,
    "steady_warm_verified": None,
    "real_time_capacity": None,
    "verified_work": None,
    "reason": "Execution alone does not establish these qualifications.",
}
THREAD_ENV = {key: "1" for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)}
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args],
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise CampaignError(f"Git check failed for {path}: {result.stderr.strip()}")
    return result.stdout.strip()


def source_guard(path, expected_commit):
    if Path(git(path, "rev-parse", "--show-toplevel")).resolve() != path:
        raise CampaignError(f"Source must be a checkout root: {path}")
    actual = git(path, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise CampaignError(f"Source commit mismatch: {path}: {actual}")
    if git(path, "status", "--porcelain", "--untracked-files=all"):
        raise CampaignError(f"Source is dirty: {path}")
    # Include ignored native extensions, which Git's clean check does not cover.
    # Bytecode is excluded and disabled for every child process.
    files = {}
    for root, directories, names in os.walk(path, followlinks=False):
        directories[:] = sorted(d for d in directories
                                if d not in {".git", "__pycache__"})
        for name in directories:
            directory = Path(root) / name
            if directory.is_symlink():
                raise CampaignError(f"Source directory symlink is unsupported: {directory}")
        for name in sorted(names):
            file = Path(root) / name
            relative = file.relative_to(path).as_posix()
            if relative == ".git":  # Linked worktrees have a .git file.
                continue
            if file.is_symlink():
                target = file.resolve()
                if not target.is_relative_to(path):
                    raise CampaignError(f"Source symlink escapes checkout: {file}")
            if file.is_file():
                files[relative] = sha256(file)
    return {"commit": actual, "files": files}


def input_guard(inputs):
    observed = {}
    for name, item in inputs.items():
        actual = sha256(item["resolved_path"])
        if actual != item["sha256"]:
            raise CampaignError(f"Input SHA256 mismatch: {name}")
        observed[name] = actual
    return observed


def assignments(values, integer_list=False):
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item or key in result:
            raise CampaignError(f"Expected one unique ARM=VALUE assignment: {value}")
        try:
            result[key] = [int(x) for x in item.split(",")] if integer_list else item
        except ValueError as error:
            raise CampaignError(f"Invalid CPU assignment: {value}") from error
    return result


def load_campaign(args):
    campaign = json.loads(args.campaign.read_text())
    if campaign.get("schema_version") != 1:
        raise CampaignError("schema_version must be 1")
    if not NAME.fullmatch(campaign.get("campaign_id", "")):
        raise CampaignError("campaign_id must be a simple filename-safe identifier")
    if campaign.get("purpose", "comparison") not in {"comparison", "diagnostic"}:
        raise CampaignError("purpose must be comparison or diagnostic")
    if campaign.get("prime_runs", 1) != 1:
        raise CampaignError("Exactly one excluded prime is required per arm")
    repeats = campaign.get("repeats", 3)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 3:
        raise CampaignError("At least three timed repeats are required")
    sources = campaign.get("sources", {})
    if not sources:
        raise CampaignError("Pinned sources are required")
    for name, source in sources.items():
        if not NAME.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source.get("commit", "")):
            raise CampaignError(f"Invalid source name or full commit: {name}")
        source["resolved_path"] = (args.sources / name).resolve()
        modules = source.get("required_modules", ["pycbc"])
        if not modules or not all(isinstance(x, str) and re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", x) for x in modules):
            raise CampaignError(f"Invalid required_modules for source {name}")
        source["required_modules"] = modules
    inputs = campaign.get("inputs", {})
    if not inputs:
        raise CampaignError("Hashed inputs are required")
    for name, item in inputs.items():
        relative = Path(item["path"])
        path = (args.inputs / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(args.inputs.resolve()):
            raise CampaignError(f"Input path must stay inside --inputs: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise CampaignError(f"Invalid input SHA256: {name}")
        item["resolved_path"] = path
    arms = campaign.get("arms", [])
    if not arms or (campaign.get("purpose", "comparison") == "comparison" and len(arms) != 4):
        raise CampaignError("Comparison campaigns require four arms; smaller campaigns must declare diagnostic purpose")
    ids = [arm.get("id", "") for arm in arms]
    if len(ids) != len(set(ids)) or not all(NAME.fullmatch(x) for x in ids):
        raise CampaignError("Arm identifiers must be unique and filename-safe")
    python_overrides = assignments(args.python)
    cpu_overrides = assignments(args.cpus, integer_list=True)
    if (set(python_overrides) | set(cpu_overrides)) - set(ids):
        raise CampaignError("An override names an unknown arm")
    for arm in arms:
        if arm.get("source") not in sources:
            raise CampaignError(f"Unknown source for {arm['id']}")
        if not isinstance(arm.get("argv"), list) or not arm["argv"] or not all(isinstance(x, str) for x in arm["argv"]):
            raise CampaignError(f"argv must be a nonempty string list: {arm['id']}")
        deadline = arm.get("deadline_seconds", 900)
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= 0:
            raise CampaignError("deadline_seconds must be positive")
        if not isinstance(arm.get("env", {}), dict) or not all(
                isinstance(key, str) and re.fullmatch(r"[A-Za-z_]\w*", key)
                for key in arm.get("env", {})):
            raise CampaignError("env must map valid environment variable names to values")
        arm["python"] = python_overrides.get(arm["id"], arm.get("python", sys.executable))
        if arm["id"] in cpu_overrides:
            arm["cpus"] = cpu_overrides[arm["id"]]
        cpus = arm.get("cpus")
        if cpus is not None and (not cpus or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in cpus)):
            raise CampaignError("cpus must contain nonnegative integer logical CPU IDs")
    profile = campaign.get("profile")
    if profile and (profile.get("arm") not in ids or not isinstance(profile.get("prefix"), list) or not profile["prefix"] or not all(isinstance(x, str) for x in profile["prefix"])):
        raise CampaignError("profile requires an existing arm and nonempty prefix argument list")
    if profile and "deadline_seconds" in profile:
        deadline = profile["deadline_seconds"]
        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= 0:
            raise CampaignError("Profile deadline_seconds must be finite and positive")
    campaign["repeats"] = repeats
    return campaign


def schedule(campaign):
    arms = campaign["arms"]
    result = [("prime", 0, arm) for arm in arms]
    for repeat in range(campaign["repeats"]):
        offset = repeat % len(arms)
        result.extend(("timed", repeat + 1, arm)
                      for arm in arms[offset:] + arms[:offset])
    if campaign.get("profile"):
        arm = next(a for a in arms if a["id"] == campaign["profile"]["arm"])
        result.append(("profile", 0, arm))
    return result


def resolve(text, context):
    def substitute(match):
        key = match.group(1)
        if key not in context:
            raise CampaignError(f"Unknown placeholder: {{{key}}}")
        return str(context[key])
    result = PLACEHOLDER.sub(substitute, text)
    if "{" in result or "}" in result:
        raise CampaignError(f"Unresolved brace in argument: {text}")
    return result


def resolved_job(campaign, arm, phase, repeat, output):
    source = campaign["sources"][arm["source"]]["resolved_path"]
    job_path = output / "runs" / f"{phase}-{repeat:02d}-{arm['id']}"
    cache = output / "cache" / arm["id"]
    context = {"python": arm["python"], "source": source,
               "output": job_path / "output", "cache": cache}
    context.update({f"input:{key}": value["resolved_path"]
                    for key, value in campaign["inputs"].items()})
    # Runtime library/toolkit paths remain inherited; scientific selectors and
    # backend/cache configuration must come from the campaign itself.
    toolkit_paths = {"CUDA_HOME", "CUDA_PATH", "CUDA_INSTALL_PATH", "CUDA_TOOLKIT_ROOT_DIR"}
    env = {key: value for key, value in os.environ.items()
           if not (key.startswith(("PYCBC_", "JAX_", "XLA_", "CUDA_"))
                   and key not in toolkit_paths)
           and key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"}}
    controls = dict(THREAD_ENV, PYTHONDONTWRITEBYTECODE="1",
                    PYCBC_JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    PYCBC_BENCHMARK_STAGES="0")
    if arm["source"] == "original":
        controls.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    controls.update({key: resolve(str(value), context)
                     for key, value in arm.get("env", {}).items()})
    # Source, private storage and timing synchronization cannot be overridden.
    controls.update(PYTHONPATH=str(source), XDG_CACHE_HOME=str(cache / "xdg"),
                    JAX_COMPILATION_CACHE_DIR=str(cache / "jax"),
                    PYCBC_JAX_CACHE_DIR=str(cache / "jax"),
                    PYCBC_CACHE_DIR=str(cache / "pycbc"),
                    CUDA_ZLIB_CACHE_DIR=str(cache / "cuda-zlib"),
                    CUDA_CACHE_PATH=str(cache / "cuda"),
                    MPLCONFIGDIR=str(cache / "matplotlib"),
                    TMPDIR=str(cache / "tmp"), TEMP=str(cache / "tmp"), TMP=str(cache / "tmp"),
                    PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                    PYCBC_BENCHMARK_SYNCHRONIZE_STAGES="0")
    env.update(controls)
    argv = [resolve(value, context) for value in arm["argv"]]
    deadline = arm.get("deadline_seconds", 900)
    if phase == "profile":
        argv = [resolve(value, context) for value in campaign["profile"]["prefix"]] + argv
        deadline = campaign["profile"].get("deadline_seconds", deadline)
    return {"path": job_path, "cache": cache, "source": source, "argv": argv,
            "env": env, "environment_controls": controls, "deadline": deadline,
            "cpus": arm.get("cpus"),
            "private_directories": [cache / name for name in
                                    ("xdg", "jax", "pycbc", "cuda-zlib", "cuda", "matplotlib", "tmp")]}


def terminate_group(process):
    """Stop descendants as well as the immediate native command."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def native_process(job):
    cpus = job["cpus"]
    if cpus is not None and not hasattr(os, "sched_setaffinity"):
        raise CampaignError("CPU affinity is unsupported on this OS; omit --cpus")
    def affinity():
        if cpus is not None:
            os.sched_setaffinity(0, cpus)
    receipt = {"argv": job["argv"], "cwd": str(job["source"]),
               "environment_controls": job["environment_controls"],
               "inherited_runtime_environment": {
                   key: value for key, value in job["env"].items()
                   if key.startswith(("PYCBC_", "JAX_", "XLA_", "CUDA_")) or key in {
                       "PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH",
                       "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONHOME", "PYTHONUSERBASE",
                   }},
               "requested_logical_cpus": cpus,
               "deadline_seconds": job["deadline"], "status": "running"}
    started = time.perf_counter()
    process = None
    try:
        with (job["path"] / "stdout-stderr.log").open("wb") as log:
            process = subprocess.Popen(job["argv"], cwd=job["source"],
                                       env=job["env"], stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       preexec_fn=affinity if cpus is not None else None)
            receipt["pid"] = process.pid
            if hasattr(os, "sched_getaffinity"):
                try:
                    receipt["observed_initial_logical_cpus"] = sorted(os.sched_getaffinity(process.pid))
                except ProcessLookupError:
                    receipt["observed_initial_logical_cpus"] = None
            write_json(job["path"] / "execution.json", receipt)
            try:
                receipt["returncode"] = process.wait(timeout=job["deadline"])
            except subprocess.TimeoutExpired as error:
                raise CampaignError("Native command exceeded its deadline") from error
            if receipt["returncode"] != 0:
                raise CampaignError(f"Native command exited {receipt['returncode']}")
            # Successful CLI return must also drain its process group.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise CampaignError("Native command left descendants running")
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["status"] = "complete"
    except BaseException as error:
        if process is not None:
            terminate_group(process)
        receipt.update(status="failed", error=str(error),
                       wall_seconds=time.perf_counter() - started)
        write_json(job["path"] / "execution.json", receipt)
        raise
    write_json(job["path"] / "execution.json", receipt)
    return receipt


def import_check(arm, source, job):
    script = (
        "import importlib,json,sys; "
        "print(json.dumps({'origins': {n: list(getattr(importlib.import_module(n), '__path__', [])) "
        "+ ([importlib.import_module(n).__file__] if getattr(importlib.import_module(n), '__file__', None) else []) "
        "for n in sys.argv[1:]}, 'python_executable': sys.executable, 'python_version': sys.version}))"
    )
    preflight = dict(job, argv=[arm["python"], "-c", script, *source["required_modules"]],
                     path=job["path"].parents[1] / "preflight" / arm["id"], deadline=60)
    preflight["path"].mkdir(parents=True)
    native_process(preflight)
    text = (preflight["path"] / "stdout-stderr.log").read_text()
    try:
        metadata = json.loads(text.strip().splitlines()[-1])
        origins = metadata["origins"]
    except (ValueError, IndexError) as error:
        raise CampaignError(f"Module origin check returned invalid JSON for {arm['id']}") from error
    for name, paths in origins.items():
        if not paths or not all(Path(p).resolve().is_relative_to(source["resolved_path"]) for p in paths):
            raise CampaignError(f"Module {name} imported outside pinned source for {arm['id']}: {paths}")
    return metadata


def run_campaign(campaign, args):
    output = args.output.resolve()
    jobs = schedule(campaign)
    plan = [{"phase": phase, "repeat": repeat, "arm": arm["id"],
             "argv": resolved_job(campaign, arm, phase, repeat, output)["argv"]}
            for phase, repeat, arm in jobs]
    sources_before = {key: source_guard(item["resolved_path"], item["commit"])
                      for key, item in campaign["sources"].items()}
    inputs_before = input_guard(campaign["inputs"])
    if any(output.is_relative_to(source["resolved_path"])
           for source in campaign["sources"].values()) or output.is_relative_to(args.inputs.resolve()):
        raise CampaignError("--output must be outside every source checkout and the input directory")
    if args.plan:
        print(json.dumps({"campaign_id": campaign["campaign_id"], "schedule": plan,
                          "qualification": UNAVAILABLE}, indent=2))
        return
    if output.exists():
        raise CampaignError("--output must not already exist; evidence is never overwritten")
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "campaign.json", json.loads(args.campaign.read_text()))
    write_json(output / "schedule.json", plan)
    write_json(output / "guards-before.json", {"sources": sources_before, "inputs": inputs_before})
    receipts = []
    summary = {"campaign_id": campaign["campaign_id"], "status": "running",
               "qualification": UNAVAILABLE, "expected_work": campaign.get("work"),
               "timing_scope": "Fresh native process launch through CLI exit and log closure; excludes guards, primes and profiles."}
    write_json(output / "summary.json", summary)
    try:
        for arm in campaign["arms"]:
            job = resolved_job(campaign, arm, "prime", 0, output)
            for directory in job["private_directories"]:
                directory.mkdir(parents=True)
            origins = import_check(arm, campaign["sources"][arm["source"]], job)
            write_json(output / f"imports-{arm['id']}.json", origins)
        for phase, repeat, arm in jobs:
            job = resolved_job(campaign, arm, phase, repeat, output)
            (job["path"] / "output").mkdir(parents=True)
            source = campaign["sources"][arm["source"]]
            if source_guard(source["resolved_path"], source["commit"]) != sources_before[arm["source"]]:
                raise CampaignError("Source content changed before native command")
            input_guard(campaign["inputs"])
            receipt = native_process(job)
            receipt.update(phase=phase, repeat=repeat, arm=arm["id"])
            if source_guard(source["resolved_path"], source["commit"]) != sources_before[arm["source"]]:
                raise CampaignError("Source content changed during native command")
            input_guard(campaign["inputs"])
            receipt["guards_passed"] = True
            receipt["output_sha256"] = {p.relative_to(job["path"] / "output").as_posix(): sha256(p)
                                         for p in sorted((job["path"] / "output").rglob("*")) if p.is_file()}
            write_json(job["path"] / "receipt.json", receipt)
            receipts.append(receipt)
            print(f"{phase} {repeat} {arm['id']}: {receipt['wall_seconds']:.6f} s", flush=True)
        sources_after = {key: source_guard(item["resolved_path"], item["commit"])
                         for key, item in campaign["sources"].items()}
        if sources_after != sources_before:
            raise CampaignError("A source changed during the campaign")
        write_json(output / "guards-after.json", {"sources": sources_after,
                                                  "inputs": input_guard(campaign["inputs"])})
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        timed = [item for item in receipts if item["phase"] == "timed"]
        with (output / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["arm", "repeat", "wall_seconds"])
            writer.writeheader()
            writer.writerows({key: item[key] for key in writer.fieldnames} for item in timed)
        summary["arms"] = {arm["id"]: {
            "samples": [item["wall_seconds"] for item in timed if item["arm"] == arm["id"]],
            "median_seconds": statistics.median(values) if summary["status"] == "complete" and len(values := [item["wall_seconds"] for item in timed if item["arm"] == arm["id"]]) == campaign["repeats"] else None,
        } for arm in campaign["arms"]}
        write_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Execute a pinned native-command campaign")
    run.add_argument("campaign", type=Path)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--sources", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--python", action="append", default=[], metavar="ARM=PATH")
    run.add_argument("--cpus", action="append", default=[], metavar="ARM=0,1")
    run.add_argument("--plan", action="store_true", help="Validate hashes and clean commits, then print the schedule without execution")
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise CampaignError(f"Controller received signal {signum}")
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        run_campaign(load_campaign(args), args)
    except (CampaignError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
