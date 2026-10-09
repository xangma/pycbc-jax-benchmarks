"""Focused runner contract checks using tiny real native-process fixtures."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from pycbc_jax_benchmarks.__main__ import main


SCRIPT = '''import os, pathlib, subprocess, sys, time
mode, input_name, output_name = sys.argv[1:]
output = pathlib.Path(output_name)
if mode in {"fail-child", "timeout-child"}:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (output / "child.pid").write_text(str(child.pid))
    if mode == "timeout-child":
        time.sleep(60)
    raise SystemExit(7)
if mode == "mutate-input":
    pathlib.Path(input_name).write_text("changed input")
if mode == "mutate-source":
    pathlib.Path(__file__).with_name("native.so").write_text("changed native extension")
(output / "result.txt").write_bytes(pathlib.Path(input_name).read_bytes())
(output / "controls.json").write_text(__import__("json").dumps({
    key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "PYTHONPATH", "JAX_COMPILATION_CACHE_DIR",
        "PYTHONDONTWRITEBYTECODE", "PYCBC_JAX_ENABLE_X64", "PYTHONNOUSERSITE",
        "PYCBC_JAX_REFERENCE_OPERATIONS", "PYCBC_GWF_CUDA_DECOMPRESS",
        "PYCBC_JAX_CACHE_DIR", "CUDA_ZLIB_CACHE_DIR", "CUDA_CACHE_PATH",
        "MPLCONFIGDIR", "TMPDIR", "TEMP", "TMP", "XLA_FLAGS", "JAX_PLATFORMS",
        "PYCBC_BENCHMARK_SYNCHRONIZE_STAGES", "CUDA_HOME")}))
'''


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "input.txt").write_text("known native fixture input\n")
        self.sources = self.root / "sources"
        self.source = self.sources / "candidate"
        self.source.mkdir(parents=True)
        (self.source / "fixture").mkdir()
        (self.source / "fixture" / "__init__.py").write_text("")
        (self.source / "run.py").write_text(SCRIPT)
        self.git("init", "-q")
        self.git("config", "user.name", "Runner fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.output = self.root / "output"
        self.manifest = self.root / "campaign.json"
        self.campaign = {
            "schema_version": 1, "campaign_id": "fixture", "purpose": "comparison",
            "sources": {"candidate": {"commit": self.git("rev-parse", "HEAD"),
                                       "required_modules": ["fixture"]}},
            "inputs": {"data": {"path": "input.txt", "sha256": hashlib.sha256(
                (self.inputs / "input.txt").read_bytes()).hexdigest()}},
            "prime_runs": 1, "repeats": 3,
            "arms": [{"id": name, "source": "candidate", "argv": [
                "{python}", "{source}/run.py", "success", "{input:data}", "{output}"],
                "deadline_seconds": 10} for name in ("a", "b", "c", "d")],
        }

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args],
                                       text=True, stderr=subprocess.STDOUT).strip()

    def invoke(self, *extra):
        self.manifest.write_text(json.dumps(self.campaign))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(["run", str(self.manifest), "--inputs", str(self.inputs),
                         "--sources", str(self.sources), "--output", str(self.output), *extra])

    def test_actual_native_fixture_rotation_guards_and_receipts(self):
        self.campaign["profile"] = {"arm": "a", "prefix": [sys.executable, "-c",
            "import os,sys; os.execvp(sys.argv[1], sys.argv[1:])"]}
        self.assertEqual(self.invoke(), 0)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertIsNone(summary["qualification"]["scientifically_qualified"])
        self.assertIsNone(summary["qualification"]["steady_warm_verified"])
        self.assertIsNone(summary["qualification"]["verified_work"])
        self.assertTrue(all(len(arm["samples"]) == 3 and arm["median_seconds"] > 0
                            for arm in summary["arms"].values()))
        with (self.output / "timings.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)  # Four primes and one profile are excluded.
        self.assertEqual([row["arm"] for row in rows], list("abcdbcdacdab"))
        self.assertEqual(json.loads((self.output / "guards-before.json").read_text()),
                         json.loads((self.output / "guards-after.json").read_text()))
        result = self.output / "runs" / "timed-01-a" / "output"
        self.assertEqual((result / "result.txt").read_bytes(),
                         (self.inputs / "input.txt").read_bytes())
        env = json.loads((result / "controls.json").read_text())
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["PYCBC_JAX_ENABLE_X64"], "1")
        self.assertEqual(env["PYTHONPATH"], str(self.source.resolve()))
        prime = json.loads((self.output / "runs" / "prime-00-a" / "receipt.json").read_text())
        timed = json.loads((self.output / "runs" / "timed-01-a" / "receipt.json").read_text())
        self.assertEqual(prime["environment_controls"]["JAX_COMPILATION_CACHE_DIR"],
                         timed["environment_controls"]["JAX_COMPILATION_CACHE_DIR"])
        self.assertFalse(self.git("status", "--porcelain"))
        self.assertEqual(self.invoke(), 1)  # Evidence cannot be overwritten.

    def test_plan_checks_without_creating_output(self):
        self.assertEqual(self.invoke("--plan"), 0)
        self.assertFalse(self.output.exists())

    def test_bad_input_hash_rejected_before_execution(self):
        self.campaign["inputs"]["data"]["sha256"] = "0" * 64
        self.assertEqual(self.invoke(), 1)
        self.assertFalse(self.output.exists())

    def test_inherited_selectors_scrubbed_and_private_storage_protected(self):
        self.campaign["purpose"] = "diagnostic"
        self.campaign["arms"] = self.campaign["arms"][:1]
        unsafe = str(self.root / "global-cache")
        self.campaign["arms"][0]["env"] = {
            "PYCBC_GWF_CUDA_DECOMPRESS": "deflate",  # Explicit selection is retained.
            "PYTHONPATH": unsafe, "PYTHONNOUSERSITE": "0", "PYTHONDONTWRITEBYTECODE": "0",
            "PYCBC_JAX_CACHE_DIR": unsafe, "CUDA_ZLIB_CACHE_DIR": unsafe,
            "MPLCONFIGDIR": unsafe, "TMPDIR": unsafe, "TEMP": unsafe, "TMP": unsafe,
            "PYCBC_BENCHMARK_SYNCHRONIZE_STAGES": "1",
        }
        contaminated = {
            "PYCBC_JAX_REFERENCE_OPERATIONS": "fft,filter", "PYCBC_GWF_CUDA_DECOMPRESS": "cpu",
            "PYCBC_JAX_CACHE_DIR": unsafe, "CUDA_ZLIB_CACHE_DIR": unsafe,
            "XLA_FLAGS": "--unknown-inherited-scientific-setting", "JAX_PLATFORMS": "cuda",
            "PYTHONPATH": unsafe, "CUDA_HOME": "/toolkit/location",
        }
        with patch.dict(os.environ, contaminated):
            self.assertEqual(self.invoke(), 0)
        output = self.output / "runs" / "timed-01-a" / "output"
        env = json.loads((output / "controls.json").read_text())
        self.assertIsNone(env["PYCBC_JAX_REFERENCE_OPERATIONS"])
        self.assertIsNone(env["XLA_FLAGS"])
        self.assertIsNone(env["JAX_PLATFORMS"])
        self.assertEqual(env["PYCBC_GWF_CUDA_DECOMPRESS"], "deflate")
        self.assertEqual(env["CUDA_HOME"], "/toolkit/location")
        self.assertEqual(env["PYTHONPATH"], str(self.source.resolve()))
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["PYCBC_BENCHMARK_SYNCHRONIZE_STAGES"], "0")
        for name in ("PYCBC_JAX_CACHE_DIR", "CUDA_ZLIB_CACHE_DIR", "CUDA_CACHE_PATH",
                     "JAX_COMPILATION_CACHE_DIR", "MPLCONFIGDIR", "TMPDIR", "TEMP", "TMP"):
            self.assertTrue(Path(env[name]).is_relative_to((self.output / "cache" / "a").resolve()))
            self.assertTrue(Path(env[name]).is_dir())
        self.assertFalse(Path(unsafe).exists())
        preflight = json.loads((self.output / "preflight" / "a" / "execution.json").read_text())
        self.assertEqual(preflight["environment_controls"]["TMPDIR"], env["TMPDIR"])

    def test_dirty_source_and_wrong_commit_rejected(self):
        (self.source / "run.py").write_text(SCRIPT + "\n# dirty\n")
        self.assertEqual(self.invoke(), 1)
        (self.source / "run.py").write_text(SCRIPT)
        self.campaign["sources"]["candidate"]["commit"] = "0" * 40
        self.assertEqual(self.invoke(), 1)
        self.assertFalse(self.output.exists())

    def test_import_outside_pinned_checkout_rejected(self):
        self.campaign["sources"]["candidate"]["required_modules"] = ["json"]
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["status"], "failed")
        self.assertFalse((self.output / "runs").exists())

    def test_native_input_change_rejected_and_not_counted(self):
        self.campaign["purpose"] = "diagnostic"
        self.campaign["arms"] = self.campaign["arms"][:1]
        self.campaign["arms"][0]["argv"][2] = "mutate-input"
        self.assertEqual(self.invoke(), 1)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["arms"]["a"]["samples"], [])

    def test_ignored_native_source_change_rejected(self):
        (self.source / ".gitignore").write_text("native.so\n")
        (self.source / "native.so").write_bytes(b"native fixture bytes")
        self.git("add", ".gitignore")
        self.git("commit", "-qm", "native fixture ignore")
        self.campaign["sources"]["candidate"]["commit"] = self.git("rev-parse", "HEAD")
        self.campaign["purpose"] = "diagnostic"
        self.campaign["arms"] = self.campaign["arms"][:1]
        self.campaign["arms"][0]["argv"][2] = "mutate-source"
        self.assertEqual(self.invoke(), 1)
        self.assertFalse(self.git("status", "--porcelain"))
        self.assertEqual(json.loads((self.output / "summary.json").read_text())["status"], "failed")

    def test_subsets_must_declare_diagnostic_and_repeats_cannot_shrink(self):
        self.campaign["arms"] = self.campaign["arms"][:1]
        self.assertEqual(self.invoke("--plan"), 1)
        self.campaign["purpose"] = "diagnostic"
        self.assertEqual(self.invoke("--plan"), 0)
        self.campaign["repeats"] = 2
        self.assertEqual(self.invoke("--plan"), 1)

    def assert_child_stopped(self, mode):
        self.campaign["purpose"] = "diagnostic"
        self.campaign["arms"] = self.campaign["arms"][:1]
        self.campaign["arms"][0]["argv"][2] = mode
        if mode == "timeout-child":
            self.campaign["arms"][0]["deadline_seconds"] = 0.3
        self.assertEqual(self.invoke(), 1)
        pid = int((self.output / "runs" / "prime-00-a" / "output" / "child.pid").read_text())
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().split()[2] == "Z":
                break  # A reparented zombie has stopped executing.
            time.sleep(0.02)
        else:
            self.fail(f"Native descendant {pid} survived cleanup")
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertIsNone(summary["arms"]["a"]["median_seconds"])
        self.assertEqual(summary["arms"]["a"]["samples"], [])

    def test_failed_native_command_cleans_process_group(self):
        self.assert_child_stopped("fail-child")

    def test_deadline_cleans_process_group(self):
        self.assert_child_stopped("timeout-child")


if __name__ == "__main__":
    unittest.main()
