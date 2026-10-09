"""Run three fresh timing jobs or one separate Nsight job; root launches it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=('samples', 'profile'), required=True)
parser.add_argument('--config', default=str(Path(__file__).with_name('config-v2.json')))
parser.add_argument('--output-root')
parser.add_argument('--affinity')
args = parser.parse_args()
config_path = Path(args.config).resolve()
config = json.loads(config_path.read_text())
root = Path(args.output_root or config['output_root']).resolve()
root.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, **config['environment'])
env['PYTHONPATH'] = config['source_root']
env['TMPDIR'] = str(root / 'tmp')
env['JAX_COMPILATION_CACHE_DIR'] = config['jax_compilation_cache_dir']
for key in ('TMPDIR', 'JAX_COMPILATION_CACHE_DIR'):
    Path(env[key]).mkdir(exist_ok=True)
entry = str(Path(__file__).with_name('entry-v2.py').resolve())
terminal = root / (args.mode + '-status.json')
record = {'status': 'running', 'pid': os.getpid(), 'cwd': config['source_root'],
          'source_revision': config['source_revision'], 'mode': args.mode,
          'stop': f'kill -TERM {os.getpid()}', 'jobs': [], 'started_ns': time.time_ns()}
child = None

def write():
    terminal.write_text(json.dumps(record, indent=2) + '\n')

def stop(*unused):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
    raise InterruptedError('Controller stopped')

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def run(label, command, timeout=None):
    global child
    log = root / (label + '.log')
    job = {'label': label, 'command': command, 'log': str(log), 'started_ns': time.time_ns()}
    record['jobs'].append(job)
    started = time.perf_counter()
    with log.open('w') as stream:
        child = subprocess.Popen(command, cwd=config['source_root'], env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        job.update(pid=child.pid, stop=f'kill -TERM -- -{child.pid}')
        write()
        try:
            job['exit_code'] = child.wait(timeout=timeout or config['timeout_sec'])
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    job.update(launch_to_exit_sec=time.perf_counter()-started, finished_ns=time.time_ns())
    write()
    if job['exit_code']:
        raise RuntimeError(f'{label} exited {job["exit_code"]}; see {log}')
    child = None
    return job


try:
    write()
    labels = ['prime', *[f'sample-{i:02d}' for i in range(1, config['samples']+1)]] if args.mode == 'samples' else ['profile']
    summaries = []
    affinity = args.affinity or config['affinity']
    assert len(affinity.split(',')) == 2 and len(set(affinity.split(','))) == 2, 'Require exactly two distinct CPU IDs'
    if args.mode == 'profile':
        prior = json.loads((root/'samples-status.json').read_text())
        assert prior['status'] == 'passed', 'Timing/prime jobs must finish before profiling'
    for label in labels:
        output = root / label
        if output.exists():
            raise RuntimeError('Refusing to reuse existing output: ' + str(output))
        output.mkdir()
        command = ['taskset', '--cpu-list', affinity,
                   'mpirun', '--bind-to', 'none', '-n', str(config['ranks']),
                   config['python'], entry, '--config', str(config_path),
                   '--source-root', config['source_root'], '--output-dir', str(output)]
        if args.mode == 'profile':
            command = ['nsys', 'profile', '--trace=cuda,nvtx,osrt', '--nvtx-domain-exclude=TSL',
                       '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
                       '--wait=all', '--force-overwrite=false', '--output', str(root/'live'), *command]
        job = run(label, command)
        ranks = [json.loads((output/f'rank-{i}.json').read_text()) for i in range(config['ranks'])]
        assert all(row['status'] == 'passed' for row in ranks)
        assert all(row['source_unchanged'] for row in ranks)
        # Root is authoritative for unique detector-time, never count both ranks.
        root_rank = ranks[0]
        by_ifo = {ifo: [] for ifo in ('H1', 'L1')}
        for row in root_rank['warmed_blocks']:
            for ifo in row['live_detectors']:
                by_ifo[ifo].append((float(row['data_start']), float(row['data_end'])))
        unique_seconds = {}
        for ifo, intervals in by_ifo.items():
            merged = []
            for begin, end in sorted(intervals):
                if merged and begin <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([begin, end])
            unique_seconds[ifo] = sum(end-begin for begin, end in merged)
        warm_epoch = root_rank['warmed_blocks'][0]['data_start']
        rank_starts = [next(row['start_ns'] for row in value['blocks'] if row['data_start']==warm_epoch) for value in ranks]
        warm_start, complete = min(rank_starts), max(value['completed_ns'] for value in ranks)
        seconds = (complete-warm_start)/1e9
        work = config['workload']['templates'] * sum(unique_seconds.values())
        summary = {'label': label, 'launch_to_exit_sec': job['launch_to_exit_sec'],
                   'warm_start_ns': warm_start, 'completed_ns': complete,
                   'warmed_suffix_sec': seconds, 'warm_data_start': warm_epoch,
                   'unique_valid_detector_seconds': unique_seconds,
                   'completed_template_detector_seconds': work,
                   'templates_per_gpu': work/seconds, 'allocated_cpu_cores': 2,
                   'templates_per_allocated_core': work/seconds/2,
                   'zero_warm_compilation': not any(event[0] >= warm_start and event[1].startswith('/jax/core/compile/') for value in ranks for event in value['compile_events']),
                   'excluded_from_timing_samples': label in ('prime', 'profile')}
        summaries.append(summary)
    if args.mode == 'profile':
        run('export', ['nsys', 'export', '--type', 'sqlite', '--output', str(root/'live.sqlite'), str(root/'live.nsys-rep')], timeout=300)
    record.update(status='passed', summaries=summaries,
                  usable_as_warmed_timing=all(row['zero_warm_compilation'] for row in summaries))
    if args.mode == 'samples':
        summaries = [row for row in summaries if row['label'] != 'prime']
        record['timing_summaries'] = summaries
        record['usable_as_warmed_timing'] = all(row['zero_warm_compilation'] for row in summaries)
        record['median_launch_to_exit_sec'] = statistics.median(row['launch_to_exit_sec'] for row in summaries)
        record['median_warmed_suffix_sec'] = statistics.median(row['warmed_suffix_sec'] for row in summaries)
    record['driver_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(entry), config_path, config_path.parent/'source-manifest.json')}
except BaseException as exc:
    record.update(status='failed', error=repr(exc))
    raise
finally:
    record['finished_ns'] = time.time_ns()
    write()
