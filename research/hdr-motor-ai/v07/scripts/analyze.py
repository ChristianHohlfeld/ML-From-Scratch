"""Summarize real camera measurements. Refuse missing provenance."""
from __future__ import annotations
import csv,collections,math,argparse,json
from pathlib import Path


def summarize(path:Path):
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows or not all(r.get('source')=='CAMERA' for r in rows):
        raise ValueError('Only nonempty camera-observed hardware records accepted')
    grouped=collections.defaultdict(list)
    for row in rows:grouped[int(row['pose'])].append(row)
    if len(grouped)<5:raise ValueError('Insufficient independent hardware poses')
    errors=[float(r['error_mm']) for r in rows]
    e=np_sort(errors)
    # Command-to-physical-settled readout is NOT latency. Need trigger/scope!
    out={'source':'REAL CAMERA MEASUREMENTS (not digital twin)', 'n_poses':len(grouped),'n_frames':len(rows),
         'rms_mm':math.sqrt(sum(v*v for v in errors)/len(errors)),
         'p95_mm':e[min(len(e)-1,math.ceil(.95*len(e))-1)],'max_mm':max(e),
         'electrical_coverage':sum(math.isfinite(float(r['watts'])) for r in rows)/len(rows),
         'latency_s':'NOT MEASURED (requires timestamp-synced trigger)',
         'energy_j':'NOT MEASURED BY FRAME CSV (use continuous T telemetry logger)',
         'accuracy_note':'Includes servo calibration, backlash, flex, optical and registration errors'}
    return out

def np_sort(xs):return sorted(xs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.csv),indent=2))
