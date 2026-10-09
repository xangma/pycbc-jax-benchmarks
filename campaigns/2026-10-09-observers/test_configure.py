"""Cheap archive/configuration checks; no application imports or jobs."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('observer_configure', HERE / 'configure.py')
CONFIGURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIGURE)


class ConfigureTests(unittest.TestCase):
    def test_frozen_archive_hashes(self):
        provenance = CONFIGURE.load(HERE / 'provenance.json')
        for name, expected in {**provenance['archived_files_sha256'],
                               **provenance['public_templates']}.items():
            self.assertEqual(CONFIGURE.sha256(HERE / name), expected, name)

    def test_token_expansion_keys_and_spaces(self):
        result = CONFIGURE.expand({'H1:${input.frame}': ['${source}/bin/tool']},
                                  {'input.frame': 'data/a b.gwf', 'source': 'source tree'})
        self.assertEqual(result, {'H1:data/a b.gwf': ['source tree/bin/tool']})
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            CONFIGURE.expand('${unknown}', {})

    def test_interpreter_symlink_spelling_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            link = Path(folder) / 'environment-python'
            link.symlink_to(sys.executable)
            self.assertEqual(CONFIGURE.executable(str(link)), str(link))

    def test_native_manifest_uses_current_files(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder).resolve()
            (source / 'pycbc').mkdir()
            (source / 'pycbc/version.py').write_text('version = "fixture"\n')
            (source / 'pycbc/core.fixture.so').write_bytes(b'new build')
            original = {'pycbc/core.cpython-312-original.so': 'old hash'}
            with patch.object(CONFIGURE, 'load', side_effect=[original, {'files': {}}]):
                result = CONFIGURE.runtime_files(source, '.fixture.so')
            self.assertEqual(result['pycbc/core.fixture.so'], hashlib.sha256(b'new build').hexdigest())
            self.assertIn('pycbc/version.py', result)
            with patch.object(CONFIGURE, 'load', return_value=original):
                with self.assertRaisesRegex(ValueError, 'Build the configured'):
                    CONFIGURE.runtime_files(source, '.missing.so')

    def test_exact_source_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder).resolve()
            (source / 'tracked.py').write_bytes(b'original')
            manifest = {'files': {'tracked.py': {'mode': '100644',
                         'sha256': hashlib.sha256(b'original').hexdigest()}}}
            good = [CONFIGURE.REVISION, CONFIGURE.TREE, '']
            with patch.object(CONFIGURE, 'git', side_effect=good), patch.object(CONFIGURE, 'load', return_value=manifest):
                self.assertEqual(CONFIGURE.verify_source(source)['tracked_files_verified'], 1)
            with patch.object(CONFIGURE, 'git', side_effect=['wrong', CONFIGURE.TREE]):
                with self.assertRaisesRegex(ValueError, 'exact archived'):
                    CONFIGURE.verify_source(source)
            with patch.object(CONFIGURE, 'git', side_effect=[CONFIGURE.REVISION, CONFIGURE.TREE, ' M tracked.py']):
                with self.assertRaisesRegex(ValueError, 'clean'):
                    CONFIGURE.verify_source(source)
            (source / 'tracked.py').write_bytes(b'changed')
            with patch.object(CONFIGURE, 'git', side_effect=good), patch.object(CONFIGURE, 'load', return_value=manifest):
                with self.assertRaisesRegex(ValueError, 'Frozen source differs'):
                    CONFIGURE.verify_source(source)

    def test_original_scientific_settings(self):
        inspiral = CONFIGURE.load(HERE / 'inspiral/config.template.json')
        live = CONFIGURE.load(HERE / 'live/config.template.json')
        self.assertEqual(inspiral['cli'][inspiral['cli'].index('--batch-size') + 1], '64')
        self.assertEqual(inspiral['input_contract']['fingerprint'],
                         '77050d2d0d0e4378034517a23418392ba983e96dee4b9d8f6d0e894a8df1d8e5')
        self.assertEqual(live['ranks'], 2)
        self.assertEqual(live['warmup_valid_blocks'], 2)
        self.assertEqual(live['args'][live['args'].index('--jax-reference-operations') + 1], 'waveform')
        self.assertEqual(live['environment']['PYCBC_BENCHMARK_SYNCHRONIZE_STAGES'], '0')
        self.assertEqual(live['environment']['CUDA_VISIBLE_DEVICES'], '${gpu}')


if __name__ == '__main__':
    unittest.main()
