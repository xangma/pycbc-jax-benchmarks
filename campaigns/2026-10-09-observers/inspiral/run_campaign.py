"""Three fresh-process samples followed by one separate whole-process Nsight run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

CHILD = None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def stop(signum, frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    raise KeyboardInterrupt('Signal ' + str(signum))


def verify(config, here):
    source = Path(config['source'])
    records = json.loads((here / 'source-manifest.json').read_text())
    for path, item in records['files'].items():
        target = source / path
        if item['mode'] == '120000':
            digest = hashlib.sha256(os.readlink(target).encode()).hexdigest()
        else:
            digest = sha(target)
        if digest != item['sha256']:
            raise RuntimeError('Frozen source changed: ' + path)
    runtime = json.loads(Path(config['runtime_manifest']).read_text())
    if 'pycbc/version.py' not in runtime or not any(path.endswith('.so') for path in runtime):
        raise RuntimeError('Require explicit generated-version and ABI extension staging hashes')
    for path, digest in runtime.items():
        if path in records['files'] or (not path.endswith('.so') and path != 'pycbc/version.py'):
            raise RuntimeError('Unexpected borrowed runtime file: ' + path)
        if sha(source / path) != digest:
            raise RuntimeError('Staged runtime file changed: ' + path)
    for key in ('frame', 'bank'):
        if sha(config[key]) != config[key + '_sha256']:
            raise RuntimeError('Input changed: ' + key)
    return {'source_manifest_sha256': sha(here / 'source-manifest.json'),
            'runtime_manifest_sha256': sha(config['runtime_manifest']),
            'runtime_files': runtime}


def main():
    global CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--reference-set', choices=('default', 'all'), default='default')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    config = json.loads(args.config.read_text())
    if config['revision'] != '366c7a955570deabaa1219d353f7847b21384698':
        raise RuntimeError('Unexpected source revision')
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'config': config,
                          'phases': ['preflight', 'prime', 'sample-1', 'sample-2', 'sample-3', 'profile', 'export', 'stats']}))
        return
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', pid=os.getpid(), pgid=os.getpgrp(), cwd=str(Path.cwd()),
                  started_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
                  stop='kill -TERM ' + str(os.getpid()), revision=config['revision'],
                  reference_set=args.reference_set, science_qualification='pending separate reference comparison',
                  publishable_performance_claim=False, jobs=[])
    status_path = output / 'status.json'
    save(status_path, status)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTEST_ADDOPTS', 'PYTHONSTARTUP', 'PYCBC_BENCHMARK_SYNCHRONIZE_STAGES'):
        env.pop(key, None)
    env.update(PYTHONPATH=config['source'], PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               CUDA_VISIBLE_DEVICES=str(config['gpu']), JAX_PLATFORMS='cuda,cpu',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               PYCBC_JAX_ENABLE_X64='1', PYCBC_BENCHMARK_SYNCHRONIZE_STAGES='0',
               PYCBC_BENCHMARK_STAGES='0', JAX_COMPILATION_CACHE_DIR=str(output / 'jax-cache'),
               PYCBC_JAX_CACHE_DIR=str(output / 'jax-cache'),
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
               JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES='-1',
               TMPDIR=str(output / 'tmp'), TMP=str(output / 'tmp'), TEMP=str(output / 'tmp'),
               XDG_CACHE_HOME=str(output / 'xdg-cache'), MPLCONFIGDIR=str(output / 'mpl-cache'))
    for directory in ('tmp', 'xdg-cache', 'mpl-cache'):
        (output / directory).mkdir()
    save(output / 'environment.json', {key: env[key] for key in (
        'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'PYCBC_JAX_ENABLE_X64',
        'PYCBC_BENCHMARK_SYNCHRONIZE_STAGES', 'JAX_COMPILATION_CACHE_DIR',
        'PYCBC_JAX_CACHE_DIR', 'TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR')})

    def call(label, command, run_dir=None):
        nonlocal status
        global CHILD
        status['source_verification'] = verify(config, here)
        run_dir = run_dir or output / label
        run_dir.mkdir(parents=True, exist_ok=True)
        command = ['taskset', '-c', str(config['cpu_affinity']), *command]
        log = run_dir / 'process.log'
        before = time.monotonic()
        code = -1
        with log.open('x') as stream:
            CHILD = subprocess.Popen(command, cwd=run_dir, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            job = dict(label=label, argv=command, cwd=str(run_dir), pid=CHILD.pid,
                       pgid=CHILD.pid, stop='kill -TERM -' + str(CHILD.pid),
                       log=str(log), start_utc=datetime.now(timezone.utc).isoformat(), status='running')
            status['current_job'] = job
            save(status_path, status)
            print(json.dumps({'event': 'start', **job}), flush=True)
            try:
                code = CHILD.wait(timeout=config['timeout_seconds'])
            except subprocess.TimeoutExpired:
                job['error'] = 'Configured process deadline expired'
            finally:
                if CHILD.poll() is None:
                    os.killpg(CHILD.pid, signal.SIGTERM)
                    try:
                        CHILD.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(CHILD.pid, signal.SIGKILL)
                        CHILD.wait()
                CHILD = None
        job.update(exit_code=code, full_process_wall_seconds=time.monotonic() - before,
                   status='passed' if code == 0 else 'failed')
        observation = run_dir / 'observation.json'
        if observation.exists():
            job['observation'] = json.loads(observation.read_text())
        status['jobs'].append(job)
        status['current_job'] = None
        save(status_path, status)
        print(json.dumps({'event': 'finish', 'label': label, 'exit_code': code,
                          'full_process_wall_seconds': job['full_process_wall_seconds']}), flush=True)
        if code:
            raise RuntimeError('Failed phase: ' + label + '; ' + str(log))
        verify(config, here)
        return job

    def search(label, *extra):
        folder = output / label
        return call(label, [config['python'], '-s', str(here / 'search_driver.py'),
                           '--config', str(args.config.resolve()), '--run-dir', str(folder),
                           '--reference-set', args.reference_set, *extra], folder)

    try:
        search('preflight', '--preflight')
        if args.preflight_only:
            status['status'] = 'preflight_passed'
        elif args.reference_set == 'all':
            search('all-original-validation')
            status['status'] = 'completed_reference_output_pending_comparison'
        else:
            search('prime')
            samples = [search('sample-' + str(index)) for index in range(1, 4)]
            search('inline-linear-diagnostic', '--decompression', 'inline_linear')
            profile = output / 'profile'
            command = [config['nsys'], 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
                       '--cpuctxsw=none', '--cuda-graph-trace=node', '--wait=all',
                       '--force-overwrite=false', '--output=' + str(profile / 'cuda'),
                       config['python'], '-s', str(here / 'search_driver.py'),
                       '--config', str(args.config.resolve()), '--run-dir', str(profile), '--profile']
            call('profile', command, profile)
            call('export', [config['nsys'], 'export', '--type=sqlite',
                            '--output=' + str(profile / 'cuda.sqlite'), str(profile / 'cuda.nsys-rep')])
            call('stats', [config['nsys'], 'stats', '--report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_pushpop_sum',
                           '--format=csv', str(profile / 'cuda.nsys-rep')])
            status['timing'] = dict(samples=[item['full_process_wall_seconds'] for item in samples],
                median_full_process_wall_seconds=statistics.median(item['full_process_wall_seconds'] for item in samples),
                application_search_seconds=[item['observation']['observed']['application_search_seconds'] for item in samples],
                timing_scope='Fresh observed public-CLI processes, warm persistent cache, includes observer imports and completed HDF output; no stage fences',
                nsight_excluded_from_samples=True)
            status['status'] = 'completed_diagnostic_pending_science_and_profile_review'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(status_path, status)


if __name__ == '__main__':
    main()
