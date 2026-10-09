"""Resolve archived observer configs without importing PyCBC or starting a job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REVISION = '366c7a955570deabaa1219d353f7847b21384698'
TREE = 'd43edf428d9bba23fc0243f41a4c66faf7d4de5f'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def executable(value):
    found = shutil.which(value)
    if not found:
        raise ValueError('Executable is unavailable: ' + value)
    # Keep interpreter/tool symlink spelling: resolving a venv Python symlink
    # to its base executable would select a different environment.
    return os.path.abspath(found)


def expand(value, tokens):
    if isinstance(value, str):
        for name, replacement in tokens.items():
            value = value.replace('${' + name + '}', str(replacement))
        if re.search(r'\$\{[^}]+\}', value):
            raise ValueError('Unresolved config token: ' + value)
        return value
    if isinstance(value, list):
        return [expand(item, tokens) for item in value]
    if isinstance(value, dict):
        return {expand(key, tokens): expand(item, tokens) for key, item in value.items()}
    return value


def verify_source(source):
    revision = git(source, 'rev-parse', 'HEAD')
    tree = git(source, 'rev-parse', 'HEAD^{tree}')
    if (revision, tree) != (REVISION, TREE):
        raise ValueError('Require the exact archived source revision and tree')
    if git(source, 'status', '--porcelain=v1', '--untracked-files=normal'):
        raise ValueError('Source must be clean, including untracked nonignored files')
    manifest = load(HERE / 'inspiral/source-manifest.json')
    for relative, item in manifest['files'].items():
        path = source / relative
        digest = (hashlib.sha256(os.readlink(path).encode()).hexdigest()
                  if item['mode'] == '120000' else sha256(path))
        if digest != item['sha256']:
            raise ValueError('Frozen source differs: ' + relative)
    return {'revision': revision, 'tree': tree,
            'tracked_files_verified': len(manifest['files']), 'clean': True}


def python_info(python):
    # Read interpreter ABI and package metadata; import no application packages.
    code = """import importlib.metadata as m, json, platform, sys, sysconfig
versions = {}
for name in ('numpy', 'scipy', 'h5py', 'jax', 'jaxlib', 'lalsuite', 'mpi4py', 'cuda-zlib'):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps(dict(python=sys.version, executable=sys.executable,
    platform=platform.platform(), extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),
    distributions=versions)))
"""
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    return json.loads(subprocess.check_output([python, '-s', '-c', code], env=env, text=True))


def runtime_files(source, suffix):
    if not suffix:
        raise ValueError('Configured interpreter has no native extension suffix')
    historical = load(HERE / 'published-runtime-files.json')
    required = [name.split('.cpython-')[0] for name in historical if name.endswith('.so')]
    for prefix in required:
        if not (source / (prefix + suffix)).is_file():
            raise ValueError('Build the configured interpreter extension: ' + prefix + suffix)
    paths = [source / 'pycbc/version.py', *sorted((source / 'pycbc').rglob('*' + suffix))]
    tracked = load(HERE / 'inspiral/source-manifest.json')['files']
    result = {}
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError('Missing or externally linked runtime file: ' + str(path))
        relative = path.relative_to(source).as_posix()
        if relative in tracked:
            raise ValueError('Runtime file unexpectedly replaces tracked source: ' + relative)
        result[relative] = sha256(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--cpus', required=True, help='Two distinct allowed CPU IDs; Inspiral uses the first')
    parser.add_argument('--nvtx-library', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='One CUDA_VISIBLE_DEVICES ordinal or UUID')
    parser.add_argument('--nsys', required=True)
    parser.add_argument('--nvcc', help='CUDA compiler; defaults to CUDACXX or nvcc on PATH')
    args = parser.parse_args()
    source, inputs, output = (path.resolve() for path in (args.source, args.inputs, args.output))
    if output.is_relative_to(HERE.parents[1]) or output.is_relative_to(source) or output.exists():
        raise ValueError('Use a new private output directory outside this repository and source')
    cpus = args.cpus.split(',')
    if len(cpus) != 2 or any(not item.isdecimal() for item in cpus) or len(set(map(int, cpus))) != 2:
        raise ValueError('--cpus requires exactly two distinct nonnegative CPU IDs')
    if not args.gpu or ',' in args.gpu:
        raise ValueError('--gpu must select one GPU')
    if not args.nvtx_library.is_file():
        raise ValueError('--nvtx-library must name an existing NVTX shared library')
    python, nsys = executable(args.python), executable(args.nsys)
    nvcc = executable(args.nvcc or os.environ.get('CUDACXX') or 'nvcc')
    provenance = load(HERE / 'provenance.json')
    for relative, expected in {**provenance['archived_files_sha256'],
                               **provenance['public_templates']}.items():
        if sha256(HERE / relative) != expected:
            raise ValueError('Archived observer asset changed: ' + relative)
    source_receipt = verify_source(source)
    resolved_inputs = {}
    for key, item in load(HERE / 'inputs.json').items():
        path = inputs / item['filename']
        if sha256(path) != item['sha256']:
            raise ValueError('Prepared input differs: ' + item['filename'])
        resolved_inputs[key] = str(path)
    interpreter = python_info(python)
    runtime = runtime_files(source, interpreter['extension_suffix'])
    output.mkdir(parents=True)
    save(output / 'runtime-files.json', runtime)
    tools = output / 'tools'
    tools.mkdir()
    # The frozen Live controller invokes the name nsys. Preserve its bytes and
    # resolve that name to the explicitly selected tool through private PATH.
    shim = tools / 'nsys'
    shim.write_text('#!/bin/sh\nexec ' + shlex.quote(nsys) + ' "$@"\n')
    shim.chmod(0o755)
    tokens = {'source': source, 'python': python, 'runtime_manifest': output / 'runtime-files.json',
              'inspiral_output': output / 'inspiral/results', 'live_output': output / 'live/results',
              'live_jax_cache': output / 'live/jax-cache', 'codec_cache': output / 'live/codec-cache',
              'nvtx_library': args.nvtx_library.resolve(), 'nvcc': nvcc, 'nsys': nsys,
              'gpu': args.gpu, 'inspiral_cpu': cpus[0], 'cpus': args.cpus,
              'tool_path': str(tools) + os.pathsep + os.environ.get('PATH', '')}
    tokens.update({'input.' + key: value for key, value in resolved_inputs.items()})
    for scope, scripts in {'inspiral': ('run_campaign.py', 'search_driver.py'),
                           'live': ('run-v2.py', 'entry-v2.py')}.items():
        folder = output / scope
        folder.mkdir()
        for name in (*scripts, 'source-manifest.json'):
            shutil.copyfile(HERE / scope / name, folder / name)
        config = expand(load(HERE / scope / 'config.template.json'), tokens)
        name = 'campaign_config.json' if scope == 'inspiral' else 'config-v2.json'
        save(folder / name, config)
        if scope == 'inspiral':
            preflight = dict(config, output=str(output / 'inspiral/preflight-results'))
            save(folder / 'campaign_preflight.json', preflight)
        else:
            (folder / 'codec-cache').mkdir()
    inspiral = [python, str(output / 'inspiral/run_campaign.py'), '--config',
                str(output / 'inspiral/campaign_config.json')]
    live = [python, str(output / 'live/run-v2.py'), '--config', str(output / 'live/config-v2.json')]
    commands = {'inspiral_plan': inspiral,
                'inspiral_preflight': [*inspiral[:2], '--config', str(output / 'inspiral/campaign_preflight.json'),
                                      '--execute', '--preflight-only'],
                'inspiral_run_and_profile': [*inspiral, '--execute'],
                'live_samples': [*live, '--mode', 'samples'],
                'live_profile_after_samples': [*live, '--mode', 'profile']}
    receipt = {'status': 'configured_not_executed', 'source': source_receipt,
               'source_path': str(source), 'inputs': resolved_inputs, 'interpreter': interpreter,
               'bindings': {'cpus_requested': cpus, 'gpu_requested': args.gpu,
                            'physical_core_topology_verified': False},
               'tool_sha256': {path: sha256(path) for path in (python, nsys, nvcc, str(args.nvtx_library.resolve()))},
               'new_runtime_files': runtime,
               'published_runtime_manifest_sha256': sha256(HERE / 'published-runtime-files.json'),
               'runtime_identity': 'This configured build, not an assertion of historical binary equivalence',
               'configured_assets_sha256': {path.relative_to(output).as_posix(): sha256(path)
                                           for path in sorted(output.rglob('*')) if path.is_file()},
               'archive_provenance_sha256': sha256(HERE / 'provenance.json'), 'commands': commands,
               'scientific_qualification': 'not established', 'fully_warm_qualification': 'not established',
               'application_imports_or_jobs_started': False}
    save(output / 'configuration.json', receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(output / 'configuration.json'),
                      'commands': {key: shlex.join(value) for key, value in commands.items()}}, indent=2))


if __name__ == '__main__':
    main()
