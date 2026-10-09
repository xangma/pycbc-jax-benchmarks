#!/usr/bin/env python3
"""Analyze duration-aware Live rank observers without executing PyCBC or JAX."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

DURATION_NAMES = {
    '/jax/core/compile/jaxpr_trace_duration': 'trace',
    '/jax/core/compile/jaxpr_to_mlir_module_duration': 'lower',
    '/jax/core/compile/backend_compile_duration': 'backend_or_cache_load',
}
TARGET_STAGES = {'coinc_resident_query_dispatch', 'coinc_resident_cluster_dispatch'}


def union(rows):
    merged = []
    for start, end in sorted(rows):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def ns(rows):
    return sum(end-start for start, end in union(rows))


def group_durations(details, lo, hi):
    grouped = collections.defaultdict(list)
    for detail in details:
        seconds = detail.get('seconds')
        if detail['name'] not in DURATION_NAMES or not isinstance(seconds, (int, float)):
            continue
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid recorded duration')
        timestamp = int(detail['monotonic_ns'])
        if not lo <= timestamp <= hi:
            continue
        fun_name = detail.get('metadata', {}).get('fun_name', '<unspecified>')
        if not isinstance(fun_name, str):
            fun_name = json.dumps(fun_name, sort_keys=True)
        grouped[(fun_name, DURATION_NAMES[detail['name']])].append(detail)
    result = []
    for (function, category), values in grouped.items():
        inferred = [(int(d['monotonic_ns'])-round(d['seconds']*1e9), int(d['monotonic_ns'])) for d in values]
        clipped = [(max(lo,a), min(hi,b)) for a,b in inferred]
        result.append({'fun_name': function, 'category': category, 'events': len(values),
                       'sum_reported_seconds': sum(d['seconds'] for d in values),
                       'max_reported_seconds': max(d['seconds'] for d in values),
                       'inferred_interval_union_seconds': ns(inferred)/1e9,
                       'inferred_overlap_with_scope_seconds': ns(clipped)/1e9,
                       'durations_starting_before_scope': sum(a<lo for a,b in inferred),
                       'thread_ids': sorted({int(d['thread_id']) for d in values})})
    return sorted(result, key=lambda row: row['sum_reported_seconds'], reverse=True)


def pair_snapshots(snapshots):
    pending = collections.defaultdict(list)
    pairs, unpaired_ends = [], []
    for row in sorted(snapshots, key=lambda row: row['monotonic_ns']):
        key = row['stage'], int(row['thread_id'])
        if row['event'] == 'start':
            pending[key].append(row)
        elif row['event'] == 'end':
            if pending[key]:
                pairs.append((pending[key].pop(), row))
            else:
                unpaired_ends.append(row)
    return pairs, sum(map(len, pending.values())), len(unpaired_ends)


def stage_ranges(path):
    pending = collections.defaultdict(list)
    result = []
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line.removeprefix('PYCBC_STAGE_EVENT '))
        key = (row['stage'], row.get('thread_id'), row.get('data_start'), row.get('data_end'))
        if row['event'] == 'start':
            pending[key].append(row)
        elif row['event'] == 'end' and pending[key]:
            start = pending[key].pop()
            if row['monotonic_ns'] > start['monotonic_ns']:
                result.append({'stage': row['stage'], 'start_ns': int(start['monotonic_ns']),
                               'end_ns': int(row['monotonic_ns']), 'thread_id': row.get('thread_id')})
    return result


def normalized_fun_name(value):
    name = str(value)
    while any(name.startswith(prefix) and name.endswith(')') for prefix in ('jit(', 'pjit(')):
        name = name[name.index('(')+1:-1]
    return name


def cache_status(before, after):
    if before is None or after is None:
        return 'missing'
    if after < before:
        return 'decreased_or_evicted'
    if after > before:
        return 'new_entries'
    return 'retained_positive_count' if before else 'unchanged_zero'


def analyze_rank(report, stages, common_start):
    details = sorted(report['compile_details'], key=lambda row: row['monotonic_ns'])
    lo, hi = int(report['cli_start_ns']), int(report['completed_ns'])
    pairs, unpaired_starts, unpaired_ends = pair_snapshots(report['jit_cache_snapshots'])
    invocations = []
    for sequence, (start, end) in enumerate(sorted(pairs, key=lambda rows: rows[0]['stage_monotonic_ns'])):
        begin, finish = int(start['stage_monotonic_ns']), int(end['stage_monotonic_ns'])
        if finish < begin:
            raise ValueError('Cache snapshot stage ends before it starts')
        observed = [d for d in details if begin <= d['monotonic_ns'] <= finish]
        cache = {}
        for function in sorted(set(start['cache_sizes']) | set(end['cache_sizes'])):
            a, b = start['cache_sizes'].get(function), end['cache_sizes'].get(function)
            cache[function] = {'before': a, 'after': b, 'delta': None if a is None or b is None else b-a,
                               'status': cache_status(a,b)}
        functions = group_durations(observed, begin, finish)
        for function, values in cache.items():
            values['matching_duration_functions'] = [row for row in functions
                if normalized_fun_name(row['fun_name']) == function]
        invocations.append({'sequence': sequence, 'stage': start['stage'],
                            'start_ns': begin, 'end_ns': finish, 'host_seconds': (finish-begin)/1e9,
                            'thread_id': int(start['thread_id']), 'in_common_suffix': begin >= common_start,
                            'straddles_suffix_boundary': begin < common_start < finish,
                            'jit_cache_functions': cache, 'snapshot_errors': [start['errors'], end['errors']],
                            'cache_notifications': dict(collections.Counter(d['name'] for d in observed if d['event_kind']=='event')),
                            'duration_functions': functions,
                            'duration_events_on_other_threads': sum(d['event_kind']=='duration' and d['thread_id']!=start['thread_id'] for d in observed),
                            'observer_overhead_seconds': (start['observer_overhead_ns']+end['observer_overhead_ns'])/1e9})
    scopes = {}
    for scope, begin in [('full',lo),('suffix',common_start)]:
        selected = [d for d in details if begin <= d['monotonic_ns'] <= hi]
        per_stage = []
        for name in sorted({row['stage'] for row in stages}):
            intervals = union((max(begin,row['start_ns']),min(hi,row['end_ns'])) for row in stages
                              if row['stage']==name and row['end_ns']>begin and row['start_ns']<hi)
            if not intervals:
                continue
            points = [d for d in selected if any(a<=d['monotonic_ns']<=b for a,b in intervals)]
            duration_rows = group_durations(points,begin,hi)
            per_stage.append({'stage': name, 'host_interval_union_seconds': ns(intervals)/1e9,
                              'cache_notifications': dict(collections.Counter(d['name'] for d in points if d['event_kind']=='event')),
                              'duration_functions': duration_rows})
        backend=[]
        cache_events=[d for d in selected if d['event_kind']=='event' and d['name'].startswith('/jax/compilation_cache/')]
        for d in selected:
            if d['name']!='/jax/core/compile/backend_compile_duration':
                continue
            finish=int(d['monotonic_ns']);start=finish-round(d['seconds']*1e9)
            events=[e for e in cache_events if e['thread_id']==d['thread_id'] and start<=e['monotonic_ns']<=finish]
            counts=collections.Counter(e['name'] for e in events)
            backend.append({'end_ns':finish,'reported_seconds':d['seconds'],
                            'fun_name':d['metadata'].get('fun_name','<unspecified>'),
                            'same_thread_cache_points_in_inferred_interval':dict(counts),
                            'cache_hit_observed':bool(counts['/jax/compilation_cache/cache_hits']),
                            'cache_miss_notification_observed':bool(counts['/jax/compilation_cache/cache_misses'])})
        scopes[scope]={'start_ns':begin,'end_ns':hi,'seconds':(hi-begin)/1e9,
                       'cache_notifications':dict(collections.Counter(d['name'] for d in selected if d['event_kind']=='event')),
                       'duration_functions':group_durations(selected,begin,hi),
                       'native_stages':per_stage,
                       'backend_events':sorted(backend,key=lambda row:row['reported_seconds'],reverse=True),
                       'backend_cache_hit_observed_count':sum(row['cache_hit_observed'] for row in backend),
                       'backend_cache_miss_notification_observed_count':sum(row['cache_miss_notification_observed'] for row in backend)}
    return {'rank':report['rank'],'status':report['status'],'scopes':scopes,'cache_stage_invocations':invocations,
            'unpaired_cache_snapshot_starts':unpaired_starts,'unpaired_cache_snapshot_ends':unpaired_ends,
            'compile_observer_overhead_seconds':report['compile_observer_overhead_ns']/1e9,
            'jit_cache_snapshot_overhead_seconds':report['jit_cache_snapshot_overhead_ns']/1e9}


def render_markdown(result):
    lines=['# Duration-aware Live baseline analysis','',
           'Reported durations are separated into tracing, lowering and backend/cache retrieval. Backend callbacks include persistent cache lookup/load; their presence alone is not evidence of cold compilation.','']
    for label,sample in result['samples'].items():
        lines += [f'## {label}','']
        for rank,row in sample['ranks'].items():
            suffix=row['scopes']['suffix']
            lines += [f'Rank {rank}: common suffix **{suffix["seconds"]:.6f} s**; observer callback overhead {row["compile_observer_overhead_seconds"]:.6f} s, cache-snapshot overhead {row["jit_cache_snapshot_overhead_seconds"]:.6f} s (full invocation).','',
                      '| Function | Category | Events | Summed reported seconds | Maximum seconds |',
                      '| --- | --- | ---: | ---: | ---: |']
            for fun in suffix['duration_functions'][:20]:
                lines.append(f'| `{fun["fun_name"]}` | {fun["category"]} | {fun["events"]} | {fun["sum_reported_seconds"]:.6f} | {fun["max_reported_seconds"]:.6f} |')
            lines += ['','| Stage | Invocations | Any module cache growth | All module counts unchanged | Host seconds |','| --- | ---: | ---: | ---: | ---: |']
            for stage in sorted(TARGET_STAGES):
                calls=[c for c in row['cache_stage_invocations'] if c['stage']==stage and c['in_common_suffix']]
                growth=sum(any(f['status']=='new_entries' for f in c['jit_cache_functions'].values()) for c in calls)
                retained=sum(all(f['status'] in ('retained_positive_count','unchanged_zero') for f in c['jit_cache_functions'].values()) for c in calls)
                lines.append(f'| `{stage}` | {len(calls)} | {growth} | {retained} | {sum(c["host_seconds"] for c in calls):.6f} |')
            lines += ['']
    lines += ['## Interpretation limits','',
              '- Cache sizes are host JIT wrapper entry counts. Growth proves new retained entries appeared; unchanged positive counts do not prove that the invoked signature hit or that no retracing/eviction occurred. Several module names can alias the same wrapper, so counts across function names must not be added.',
              '- Cache snapshots bracket native host marker times. Other threads may mutate caches within the same interval. Duration events on other threads and snapshot errors are preserved.',
              '- Callback durations may nest. Summed durations are not additive wall-time attribution. Inferred intervals end at callback receipt; notification/dispatch latency shifts their boundaries slightly. Boundary-crossing durations are recorded.',
              '- Same-thread cache-hit points inside a backend interval support retrieval/load on that call; missing points do not prove compilation. Cache-miss notifications are eligible write attempts, not necessarily completed writes.',
              '- Host phase overlap does not establish causal ownership of asynchronous device work. No stage completion fences or source changes are introduced by this analyzer.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    directories=sorted({p.parent for p in args.input.rglob('rank-0.json')})
    if not directories:
        raise SystemExit('No completed rank-0.json reports found; fetch observer receipts first')
    controller={}
    for path in args.input.glob('*-status.json'):
        for row in json.loads(path.read_text()).get('summaries',[]):
            controller[row['label']]=row
    samples={};inputs=[]
    for directory in directories:
        paths=[directory/f'rank-{rank}.json' for rank in [0,1]]
        reports=[json.loads(p.read_text()) for p in paths]
        for report in reports:
            if report['status']!='passed' or not report['source_unchanged'] or report['stage_synchronization']:
                raise ValueError(f'{directory.name}: observer contract failed')
            if report.get('observer_version')!='diagnostic-v3-duration-details-jit-cache-counts':
                raise ValueError('Require the duration-aware/cache-count observer')
        label=directory.name
        common_start=int(controller[label]['warm_start_ns']) if label in controller else min(int(r['warm_start_ns']) for r in reports)
        samples[label]={'common_suffix_start_ns':common_start,
                        'boundary_source':'controller common valid-data epoch' if label in controller else 'minimum of the two per-rank warm starts; controller receipt unavailable',
                        'ranks':{str(r['rank']):analyze_rank(r,stage_ranges(directory/f'rank-{r["rank"]}.stages.jsonl'),common_start) for r in reports}}
        inputs.extend({'file':p.relative_to(args.input).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths)
    result={'samples':samples,'input_rank_reports':inputs,'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'duration-cache-analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (args.output/'findings.md').write_text(render_markdown(result))
    print(json.dumps({'samples':list(samples),'ranks_per_sample':2,'output_files':['duration-cache-analysis.json','findings.md']}))


if __name__=='__main__':main()
