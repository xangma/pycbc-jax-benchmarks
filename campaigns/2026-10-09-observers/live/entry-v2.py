"""Observe one unchanged dedicated Live CLI invocation per MPI rank."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import threading
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--source-root')
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
config_path = Path(args.config).resolve()
config = json.loads(config_path.read_text())
source_root = Path(args.source_root or config['source_root']).resolve()
output = Path(args.output_dir).resolve()
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK', '-1')))
assert rank in (0, 1), 'Exactly two MPI ranks are required'
assert os.environ.get('PYCBC_BENCHMARK_STAGES') == '1'
assert os.environ.get('PYCBC_BENCHMARK_SYNCHRONIZE_STAGES') == '0'
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(source_root))
manifest = json.loads((config_path.parent / 'source-manifest.json').read_text())
assert manifest['revision'] == config['source_revision']
assert manifest['tree'] == config['source_tree']

def verify_source():
    for relative, expected in manifest['git_blob_sha1'].items():
        data = (source_root / relative).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise RuntimeError('Frozen source mismatch: ' + relative)

def verify_inputs():
    actual = {}
    for filename, expected in config['input_sha256'].items():
        digest = hashlib.sha256()
        with Path(filename).open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        actual[filename] = digest.hexdigest()
        if actual[filename] != expected:
            raise RuntimeError('Frozen input mismatch: ' + filename)
    return actual

verify_source()
input_hashes = verify_inputs()
raw_events, compile_events, listeners = [], [], []
clock = {}
observer_install_started = False


def install_observers():
    global observer_install_started
    if observer_install_started:
        return
    observer_install_started = True
    jax = sys.modules.get('jax')
    if jax is None:
        raise RuntimeError('JAX absent at first native stage marker')
    def event(name, **metadata):
        if name.startswith('/jax/compilation_cache/'):
            compile_events.append((time.perf_counter_ns(), name))
    def duration(name, seconds, **metadata):
        if name in ('/jax/core/compile/jaxpr_trace_duration',
                    '/jax/core/compile/jaxpr_to_mlir_module_duration',
                    '/jax/core/compile/backend_compile_duration'):
            compile_events.append((time.perf_counter_ns(), name))
    jax.monitoring.register_event_listener(event)
    jax.monitoring.register_event_duration_secs_listener(duration)
    listeners.extend((event, duration))
    library = ctypes.CDLL(config['nvtx_library'])
    mark = library.nvtxMarkA
    mark.argtypes, mark.restype = [ctypes.c_char_p], None
    clock.update(monotonic_ns=time.perf_counter_ns(), wall_time_ns=time.time_ns(), pid=os.getpid(), rank=rank)
    payload = json.dumps({'kind': 'PYCBC_FRESH_CLOCK', **clock}, separators=(',', ':'))
    mark(payload.encode())
    clock['marker_return_ns'] = time.perf_counter_ns()
    clock['nvtx_marker_emitted'] = True


class StageRecorder:
    """Capture complete native print arguments, independently of newlines."""
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.RLock()
    def write(self, value):
        # Native stage_event prints one complete prefix+JSON argument. Its
        # newline is a separate write and can follow another thread's event.
        # Retain malformed stage-bearing writes; JSON validation fails later.
        with self.lock:
            result = self.stream.write(value)
            if 'PYCBC_STAGE_EVENT ' in value:
                install_observers()
                raw_events.append(value.removeprefix('PYCBC_STAGE_EVENT '))
            return result
    def __getattr__(self, name):
        return getattr(self.stream, name)


script = source_root / 'bin/pycbc_live_jax'
command = [str(script), *config['args'], '--output-path', str(output / 'outputs')]
old_argv, old_stderr = sys.argv, sys.stderr
report = {'status': 'running', 'rank': rank, 'pid': os.getpid(),
          'source_revision': config['source_revision'], 'source_tree': config['source_tree'],
          'command': command, 'stage_synchronization': False,
          'reference_operations': ['waveform'], 'input_sha256': input_hashes}
started = time.perf_counter_ns()
try:
    sys.argv = command
    sys.stderr = StageRecorder(old_stderr)
    namespace = runpy.run_path(str(script), run_name='__main__')
    returned = time.perf_counter_ns()
    jax = sys.modules['jax']
    # One completion boundary after native return, never at an internal stage.
    jax.block_until_ready(jax.live_arrays(platform='gpu'))
    jax.effects_barrier()
    sys.modules['mpi4py.MPI'].COMM_WORLD.Barrier()
    completed = time.perf_counter_ns()
    records = [json.loads(line) for line in raw_events]
    assert records and not any(row.get('synchronized') for row in records)
    starts = {row['data_start']: row for row in records
              if row['stage'] == 'block' and row['event'] == 'start'}
    blocks = []
    for row in records:
        if row['stage'] != 'block' or row['event'] != 'end':
            continue
        begin = starts[row['data_start']]
        blocks.append({'data_start': row['data_start'], 'data_end': row['data_end'],
                       'start_ns': begin['monotonic_ns'], 'end_ns': row['monotonic_ns'],
                       'live_detectors': row.get('live_detectors', [])})
    valid = [row for row in blocks if set(row['live_detectors']) == {'H1', 'L1'}]
    warm = valid[config['warmup_valid_blocks']:]
    assert warm, 'No warmed valid blocks remain'
    warm_start = warm[0]['start_ns']
    warm_compiles = [row for row in compile_events if row[0] >= warm_start]
    compiled = [row for row in warm_compiles if row[1].startswith('/jax/core/compile/')]
    origins, origin_errors = {}, []
    for name, module in sorted(sys.modules.items()):
        if name != 'pycbc' and not name.startswith('pycbc.'):
            continue
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(source_root):
            origin_errors.append({'module': name, 'path': str(path)})
        origins[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert not origin_errors, origin_errors
    providers = sorted(name for name in sys.modules if name == 'diffgw' or name.startswith('diffgw.'))
    assert not providers, 'Unexpected optional waveform provider import'
    import numpy as np
    bank = namespace['bank']
    fields = [key for key in ('mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
                              'spin2x', 'spin2y', 'spin2z') if key in bank.table.dtype.names]
    assert {'mass1', 'mass2'} <= set(fields)
    distinct = len(np.unique(np.asarray(bank.table)[fields]))
    assert len(bank) == distinct == config['workload']['templates']
    dtypes = sorted({dtype for row in records for dtype in row.get('signal_dtypes', [])})
    if rank == 1:
        assert dtypes == ['complex64'], dtypes
    verify_source()
    assert verify_inputs() == input_hashes
    report.update(status='passed', cli_start_ns=started, cli_return_ns=returned,
                  completed_ns=completed, native_cli_completed_sec=(completed-started)/1e9,
                  warm_start_ns=warm_start, warm_completed_sec=(completed-warm_start)/1e9,
                  blocks=blocks, warmed_blocks=warm, compile_events=compile_events, warm_compile_events=warm_compiles,
                  zero_warm_compilation=not compiled, signal_dtypes=dtypes,
                  templates=len(bank), distinct_templates=distinct, loaded_pycbc=origins,
                  optional_provider_modules=providers, versions={'jax': jax.__version__,
                  'numpy': np.__version__, 'python': sys.version}, clock_marker=clock,
                  source_unchanged=True)
except BaseException as exc:
    report.update(status='failed', error=repr(exc))
    traceback.print_exc(file=old_stderr)
    raise
finally:
    sys.argv, sys.stderr = old_argv, old_stderr
    # One invocation per process; observers end with process exit.
    (output / f'rank-{rank}.raw-stage-writes.json').write_text(json.dumps(raw_events, indent=2)+'\n')
    (output / f'rank-{rank}.stages.jsonl').write_text('\n'.join(raw_events) + '\n')
    (output / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
