#!/usr/bin/env python3
"""Compact three-sample observer statistics; exclude prime/profile timings."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

CATEGORIES = ('trace', 'lower', 'backend_or_cache_load')


def normalized(value):
    name = str(value)
    while any(name.startswith(prefix) and name.endswith(')') for prefix in ('jit(', 'pjit(')):
        name = name[name.index('(')+1:-1]
    return name


def stats(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values), 'samples': values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--expected-samples', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    labels = sorted(label for label in data['samples'] if label.startswith('sample-'))
    if len(labels) != args.expected_samples:
        raise ValueError(f'Expected {args.expected_samples} completed samples, found {labels}')
    controller = json.loads(args.controller.read_text()) if args.controller else {}
    controller_rows = {row['label']: row for row in controller.get('summaries', [])}
    result = {'samples': labels, 'scope': 'common post-two-valid-block suffix; not proven fully warmed',
              'duration_interpretation': 'Summed reported callback durations; categories/functions may nest and are not additive wall-time attribution. Percentages use totals within the same recorded category.',
              'input_analysis_sha256': hashlib.sha256(args.analysis.read_bytes()).hexdigest(), 'ranks': {}}
    completed = [controller_rows[label]['warmed_suffix_sec'] if label in controller_rows else
                 max(rank['scopes']['suffix']['seconds'] for rank in data['samples'][label]['ranks'].values())
                 for label in labels]
    result['common_completed_suffix_seconds'] = stats(completed)
    if all(label in controller_rows for label in labels):
        result['launch_to_exit_seconds'] = stats([controller_rows[label]['launch_to_exit_sec'] for label in labels])
        result['zero_warm_compilation'] = [controller_rows[label]['zero_warm_compilation'] for label in labels]
    for rank in sorted(data['samples'][labels[0]]['ranks']):
        rows = [data['samples'][label]['ranks'][rank] for label in labels]
        scopes = [row['scopes']['suffix'] for row in rows]
        functions = sorted({normalized(fun['fun_name']) for scope in scopes for fun in scope['duration_functions']})
        category_totals = {category: [sum(fun['sum_reported_seconds'] for fun in scope['duration_functions'] if fun['category']==category) for scope in scopes]
                           for category in CATEGORIES}
        function_rows = []
        for function in functions:
            entry = {'fun_name': function, 'categories': {}}
            for category in CATEGORIES:
                durations=[];counts=[];shares=[]
                for index,scope in enumerate(scopes):
                    values=[fun for fun in scope['duration_functions'] if normalized(fun['fun_name'])==function and fun['category']==category]
                    value=sum(fun['sum_reported_seconds'] for fun in values)
                    durations.append(value);counts.append(sum(fun['events'] for fun in values))
                    shares.append(value/category_totals[category][index] if category_totals[category][index] else 0)
                entry['categories'][category]={'reported_seconds':stats(durations),'event_counts':stats(counts),
                                                'fraction_of_recorded_category_seconds':stats(shares)}
            entry['median_summed_category_seconds']=sum(entry['categories'][category]['reported_seconds']['median'] for category in CATEGORIES)
            function_rows.append(entry)
        function_rows.sort(key=lambda row:row['median_summed_category_seconds'],reverse=True)
        all_calls=[[call for call in row['cache_stage_invocations'] if call['in_common_suffix']] for row in rows]
        stages=sorted({call['stage'] for calls in all_calls for call in calls})
        cache_rows=[]
        stage_rows=[]
        for stage in stages:
            calls_by_sample=[[call for call in calls if call['stage']==stage] for calls in all_calls]
            stage_rows.append({'stage':stage,'invocation_counts':stats([len(calls) for calls in calls_by_sample]),
                               'host_seconds':stats([sum(call['host_seconds'] for call in calls) for calls in calls_by_sample])})
            active=sorted({name for calls in calls_by_sample for call in calls for name,value in call['jit_cache_functions'].items()
                           if value['matching_duration_functions'] or value['status']=='new_entries'})
            for function in active:
                counters=[];growth=[]
                for calls in calls_by_sample:
                    values=[call['jit_cache_functions'][function] for call in calls if function in call['jit_cache_functions']]
                    counters.append(collections.Counter(value['status'] for value in values))
                    growth.append(sum(max(0,value['delta'] or 0) for value in values))
                cache_rows.append({'stage':stage,'function':function,
                                   'calls_with_new_entries':stats([c['new_entries'] for c in counters]),
                                   'calls_with_unchanged_positive_count':stats([c['retained_positive_count'] for c in counters]),
                                   'positive_cache_entry_deltas':stats(growth),
                                   'calls_with_decreased_count':stats([c['decreased_or_evicted'] for c in counters])})
        result['ranks'][rank]={'functions':function_rows,'category_reported_seconds':{key:stats(value) for key,value in category_totals.items()},
                               'native_cache_stages':stage_rows,'cache_growth':cache_rows,
                               'suffix_cache_hit_notifications':stats([scope['cache_notifications'].get('/jax/compilation_cache/cache_hits',0) for scope in scopes]),
                               'suffix_cache_miss_notifications':stats([scope['cache_notifications'].get('/jax/compilation_cache/cache_misses',0) for scope in scopes]),
                               'compile_observer_overhead_full_seconds':stats([row['compile_observer_overhead_seconds'] for row in rows]),
                               'cache_snapshot_overhead_full_seconds':stats([row['jit_cache_snapshot_overhead_seconds'] for row in rows])}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'three-sample-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# Baseline: three completed diagnostic samples','',
           'Prime and the separate profile are excluded. Each sample starts a fresh process and uses the primed persistent cache. The suffix is not a proved fully warmed run.','',
           'Common completed suffix: **{median:.6f} s** median, **{min:.6f}–{max:.6f} s** range.'.format(**result['common_completed_suffix_seconds']),'']
    if 'launch_to_exit_seconds' in result:
        lines += ['Launch-to-exit: **{median:.6f} s** median, **{min:.6f}–{max:.6f} s** range.'.format(**result['launch_to_exit_seconds']),'']
    for rank,row in result['ranks'].items():
        lines += [f'## Rank {rank}','',
                  'Seconds below are medians with min–max ranges across the same three suffixes. They sum reported callbacks; overlapping/nested durations prevent treating the totals as additive wall-time attribution.','',
                  '| Function | Trace s | Lower s | Backend/cache load s | Trace/lower/backend events |',
                  '| --- | ---: | ---: | ---: | --- |']
        for function in row['functions'][:12]:
            fields=[];counts=[]
            for category in CATEGORIES:
                values=function['categories'][category]
                fields.append('{median:.3f} ({min:.3f}–{max:.3f})'.format(**values['reported_seconds']))
                counts.append('{median:g} ({min:g}–{max:g})'.format(**values['event_counts']))
            lines.append('| `'+function['fun_name']+'` | '+' | '.join(fields)+' | '+' / '.join(counts)+' |')
        lines += ['','| Cache stage / function | Calls adding entries, median (range) | Entry increase, median (range) | Calls retaining positive count, median (range) |',
                  '| --- | ---: | ---: | ---: |']
        for cache in row['cache_growth']:
            values=[cache[key] for key in ('calls_with_new_entries','positive_cache_entry_deltas','calls_with_unchanged_positive_count')]
            lines.append('| `'+cache['stage']+'` / `'+cache['function']+'` | '+' | '.join('{median:g} ({min:g}–{max:g})'.format(**value) for value in values)+' |')
        hit=row['suffix_cache_hit_notifications'];miss=row['suffix_cache_miss_notifications']
        lines += ['',f'Persistent cache notifications: hits {hit["samples"]}; misses {miss["samples"]}. A miss notification is an eligible cache write attempt, not proof of a completed write. Unchanged JIT entry counts do not establish per-call hits.','']
    lines += ['The JSON includes category shares, all function rows, observer overhead, stage host times and exact per-sample values. No source change or GPU job is performed by this analysis.','']
    (args.output/'three-sample-summary.md').write_text('\n'.join(lines))
    print(json.dumps({'samples':labels,'suffix_seconds':result['common_completed_suffix_seconds'],
                      'launch_to_exit_seconds':result.get('launch_to_exit_seconds')},indent=2))


if __name__=='__main__':main()
