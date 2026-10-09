#!/usr/bin/env python3
"""Summarize gaps in recorded kernel/copy activity from a portable profile."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def merge(rows):
    result=[]
    for a,b in sorted(rows):
        if b<=a: continue
        if result and a<=result[-1][1]: result[-1][1]=max(b,result[-1][1])
        else: result.append([a,b])
    return result


def union_ns(rows):
    return sum(b-a for a,b in merge(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    d=json.loads(gzip.decompress(args.input.read_bytes()))
    out={'definition':'Complement of selected application recorded kernel+copy intervals; not proof that the whole device is idle. Host-stage and monitor overlap is temporal, not causal. Memsets are excluded.', 'scopes':{}}
    for name in ['search','warm']:
        lo,hi=d['windows'][name]['start_ns'],d['windows'][name]['end_ns']
        active=merge((max(lo,r[0]),min(hi,r[1])) for r in d['gpu'] if r[1]>lo and r[0]<hi)
        gaps=[];last=lo
        for a,b in active:
            if a>last:gaps.append([last,a])
            last=b
        if last<hi:gaps.append([last,hi])
        longest=[]
        for a,b in sorted(gaps,key=lambda r:r[1]-r[0],reverse=True)[:10]:
            host=defaultdict(list);points=defaultdict(int)
            for x,y,rank,phase,source in d['host']:
                if x==y:
                    if a<=x<b: points[rank]+=1
                elif y>a and x<b: host[(rank,phase)].append((max(a,x),min(b,y)))
            longest.append({'start_ns':a,'end_ns':b,'duration_ns':b-a,'host_overlap_ns':[{'rank':r,'phase':p,'union_ns':union_ns(v)} for (r,p),v in sorted(host.items())], 'monitor_points_by_rank':dict(points)})
        out['scopes'][name]={'scope_ns':hi-lo,'recorded_activity_union_ns':union_ns(active),'gap_count':len(gaps),'gap_union_ns':sum(b-a for a,b in gaps), 'gaps_at_least_ns':{str(t):{'count':sum(b-a>=t for a,b in gaps),'sum_ns':sum(b-a for a,b in gaps if b-a>=t)} for t in [100000,1000000,10000000,100000000]},'per_rank_recorded_activity_union_ns':{rank:union_ns((max(lo,r[0]),min(hi,r[1])) for r in d['gpu'] if r[2]==rank and r[1]>lo and r[0]<hi) for rank in d['processes']},'longest_gaps':longest}
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
