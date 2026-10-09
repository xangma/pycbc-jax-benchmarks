"""Run the frozen public search with passive observation and optional NVTX."""
import argparse
from contextlib import contextmanager
import ctypes
import ctypes.util
import functools
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--decompression', choices=('device_linear', 'inline_linear'))
    parser.add_argument('--reference-set', choices=('default', 'all'), default='default')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = Path(config['source']).resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(status='running', pid=os.getpid(), source=str(source),
                  revision=config['revision'], profile=args.profile,
                  intrusive_synchronization=False, spans=[], observed={},
                  reference_set=args.reference_set,
                  science_qualification='not established by this profile')
    previous = tuple(sys.meta_path)

    class SourceGuard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('diffgw', 'jaxwave', 'torchwave'):
                raise ImportError('This campaign holds waveforms to stored LAL templates: ' + fullname)
            if fullname != 'pycbc' and not fullname.startswith('pycbc.'):
                return None
            for finder in previous:
                method = getattr(finder, 'find_spec', None)
                spec = method(fullname, path, target) if method else None
                if spec is None:
                    continue
                locations = ([spec.origin] if spec.origin not in (None, 'built-in', 'frozen') else [])
                locations += list(spec.submodule_search_locations or ())
                if any(not Path(item).resolve().is_relative_to(source) for item in locations):
                    raise ImportError('PyCBC import escaped frozen snapshot: ' + fullname + ': ' + repr(locations))
                return spec
            return None

    sys.path.insert(0, str(source))
    sys.meta_path.insert(0, SourceGuard())
    os.environ['PYCBC_BENCHMARK_SYNCHRONIZE_STAGES'] = '0'
    os.environ['PYCBC_BENCHMARK_STAGES'] = '1' if args.profile else '0'
    nvtx = None
    if args.profile:
        candidates = [config.get('nvtx_library'), ctypes.util.find_library('nvToolsExt'),
                      'libnvToolsExt.so.1', 'libnvToolsExt.so']
        for name in filter(None, candidates):
            try:
                nvtx = ctypes.CDLL(name)
                nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
                nvtx.nvtxRangePop.argtypes = []
                break
            except (OSError, AttributeError):
                nvtx = None
        if nvtx is None:
            raise RuntimeError('NVTX library unavailable; set nvtx_library after read-only admission')

    @contextmanager
    def span(name, **metadata):
        before = time.monotonic_ns()
        if nvtx is not None:
            nvtx.nvtxRangePushA(name.encode())
        try:
            yield
        finally:
            if nvtx is not None:
                nvtx.nvtxRangePop()
            if args.profile:
                report['spans'].append(dict(name=name, start_ns=before,
                                           end_ns=time.monotonic_ns(),
                                           thread_id=threading.get_native_id(),
                                           timing_scope='host submission; no device fence',
                                           **metadata))

    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*values, **kwargs):
            with span(label):
                return original(*values, **kwargs)
        setattr(owner, name, call)

    try:
        with span('startup_imports'):
            import h5py
            import numpy as np
            import jax
            import jaxlib
            import lal
            import lalsimulation
            import lalframe
            import pycbc
            from pycbc import scheme, strain, psd
            from pycbc.filter import matchedfilter_jax as mf
            from pycbc.waveform import bank_jax
            from pycbc.vetoes import chisq_jax
            from pycbc.events.eventmgr_jax import JAXEventManager
        devices = jax.devices('gpu')
        if len(devices) != 1:
            raise RuntimeError('Exactly one visible GPU required: ' + repr(devices))
        report['runtime'] = dict(python=sys.version, numpy=np.__version__,
                                 jax=jax.__version__, jaxlib=jaxlib.__version__,
                                 lal=getattr(lal, '__version__', None),
                                 lalsimulation=getattr(lalsimulation, '__version__', None),
                                 lalframe=getattr(lalframe, '__version__', None),
                                 device_kind=devices[0].device_kind,
                                 affinity=sorted(os.sched_getaffinity(0)))
        if args.preflight:
            for key in ('bank', 'frame'):
                path = Path(config[key])
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for data in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(data)
                if digest.hexdigest() != config[key + '_sha256']:
                    raise RuntimeError('Input hash mismatch: ' + key)
            with h5py.File(config['bank']) as bank:
                fields = ('mass1', 'mass2', 'spin1z', 'spin2z')
                values = np.column_stack([bank[field][()] for field in fields])
                if len(values) != 1536 or len(np.unique(values, axis=0)) != 1536:
                    raise RuntimeError('Expected 1536 distinct physical templates')
                if 'compressed_waveforms' not in bank:
                    raise RuntimeError('Compressed waveform bank missing')
        original_segments = strain.StrainSegments.fourier_segments
        @functools.wraps(original_segments)
        def segments(self, *values, **kwargs):
            with span('segment_fft'):
                result = original_segments(self, *values, **kwargs)
            intervals = sorted((int(item.cumulative_index),
                                int(item.cumulative_index + item.analyze.stop - item.analyze.start))
                               for item in result)
            merged = []
            for lo, hi in intervals:
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            valid = sum(hi - lo for lo, hi in merged) / float(self.sample_rate)
            report['observed'].update(segment_count=len(result),
                sample_rate=float(self.sample_rate), signal_dtypes=sorted({str(item.dtype) for item in result}),
                valid_sample_intervals=merged, valid_detector_seconds=valid)
            if len(result) != 5 or valid != 1904 or float(self.sample_rate) != 2048:
                raise RuntimeError('Search geometry differs from frozen workload')
            if report['observed']['signal_dtypes'] != ['complex64']:
                raise RuntimeError('Search precision differs from complex64')
            return result
        strain.StrainSegments.fourier_segments = segments
        original_batch = mf.process_batch_inspiral_jax
        completed = []
        @functools.wraps(original_batch)
        def batch(*values, **kwargs):
            ids = list(values[2] if len(values) > 2 else kwargs['batch_tnums'])
            with span('search_batch', templates=len(ids), first=ids[0], last=ids[-1]):
                result = original_batch(*values, **kwargs)
            completed.extend(int(index) for index in ids)
            return result
        mf.process_batch_inspiral_jax = batch
        if args.profile:
            for owner, name, label in (
                (strain, 'from_cli', 'conditioning_and_frame_io'),
                (psd, 'associate_psds_to_segments', 'psd'),
                (bank_jax, 'execute_batch_decompression_jax', 'template_decompression'),
                (mf, 'live_template_norms_jax', 'template_norms'),
                (mf, 'batched_matched_filter_and_cluster_jax', 'matched_filter_cluster'),
                (chisq_jax, 'batch_power_chisq_jax', 'point_chisq'),
                (JAXEventManager, 'finalize_events', 'event_finalize'),
                (JAXEventManager, 'write_events', 'hdf_output'),
            ):
                wrap(owner, name, label)
        operations = ('waveform' if args.reference_set == 'default'
                      else ','.join(sorted(scheme.JAX_REFERENCE_OPERATIONS)))
        cli = [str(source / 'bin/pycbc_inspiral_jax')] + list(config['cli'])
        if args.decompression:
            cli[cli.index('--waveform-decompression-method') + 1] = args.decompression
        for flag, value in (('--frame-files', config['frame']), ('--bank-file', config['bank']),
                            ('--output', str(run_dir / 'triggers.hdf')),
                            ('--jax-reference-operations', operations)):
            cli.extend([flag, value])
        report['argv'] = cli
        report['reference_operations'] = operations.split(',')
        if not args.preflight:
            sys.argv = cli
            with span('public_search'):
                runpy.run_path(cli[0], run_name='__main__')
            if sorted(completed) != list(range(1536)):
                raise RuntimeError('Search did not complete each of 1536 templates exactly once')
            report['observed'].update(completed_templates=len(completed),
                completed_template_seconds=len(completed) * report['observed']['valid_detector_seconds'])
            with h5py.File(run_dir / 'triggers.hdf') as output:
                report['observed']['triggers'] = len(output['H1/snr'])
                for field in ('run_time', 'setup_time_fraction'):
                    path = 'H1/search/' + field
                    report['observed'][field] = float(np.asarray(output[path][()]).ravel()[0])
                report['observed']['application_search_seconds'] = (
                    report['observed']['run_time'] * (1 - report['observed']['setup_time_fraction']))
                report['application_timer_scope'] = 'Existing host HDF run_time*(1-setup_time_fraction); excludes setup and HDF write; no added fence, so queued device/output work may extend beyond this timer. This is not a synchronized device-compute measurement.'
        origins = {}
        for name, module in list(sys.modules.items()):
            if name == 'pycbc' or name.startswith('pycbc.'):
                origin = getattr(module, '__file__', None)
                if origin:
                    if not Path(origin).resolve().is_relative_to(source):
                        raise RuntimeError('Loaded PyCBC origin escaped snapshot: ' + name)
                    origins[name] = origin
        report['pycbc_origins'] = origins
        report['provider_imports'] = [name for name in sys.modules
                                      if name.split('.')[0] in ('diffgw', 'jaxwave', 'torchwave')]
        report['status'] = 'preflight_passed' if args.preflight else 'completed_diagnostic'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['driver_wall_seconds'] = time.monotonic() - started
        write(run_dir / 'observation.json', report)


if __name__ == '__main__':
    main()
