"""Controlled synthetic inverse experiment, with out-of-distribution excitation.

An acoustic inverse fit of the same simulator is not evidence of world topology,
unique 3D anatomy, or acoustic energy efficiency.
"""
import json,time
from pathlib import Path
import numpy as np
from waveform.core import Shape,signature
from waveform.coupled import Wall,simulate,fit_shape

CONFIG={'steps':950,'nx':24,'ny':13,'nz':13,'nphi':12,'frames':1}
SHAPES=[((.70,1.25,.85,1.30),(.18,-.12)),((1.33,.74,1.20,.73),(-.20,.16)),((.81,1.31,1.23,.79),(.12,.20))]

def mse(a,b):return float(np.mean((a-b)**2))

def main():
    out=[];started=time.perf_counter()
    for i,(controls,lobes) in enumerate(SHAPES):
        real=Shape(controls=controls,lobes=lobes);initial=Shape(controls=(1.,1.,1.,1.))
        target=signature(simulate(real,excitation='chirp',**CONFIG))
        fit=fit_shape(target,initial=initial,steps=CONFIG['steps'],nx=CONFIG['nx'],ny=CONFIG['ny'],nz=CONFIG['nz'],nphi=CONFIG['nphi'],max_evals=48)
        held=signature(simulate(real,excitation='multitone',**CONFIG))
        bef=mse(signature(simulate(initial,excitation='multitone',**CONFIG)),held)
        aft=mse(signature(simulate(fit['shape'],excitation='multitone',**CONFIG)),held)
        rigid=mse(signature(simulate(fit['shape'],wall=Wall(enabled=False),excitation='multitone',**CONFIG)),held)
        item={'case':i,'target_controls':controls,'target_lobes':lobes,'fit_lobes':fit['shape'].lobes,'fit_controls':fit['shape'].controls,'evals':fit['evals'],
              'training_before':fit['before'],'training_after':fit['after'],
              'unseen_signal_before':bef,'unseen_signal_after':aft,
              'heldout_gain_percent':100*(1-aft/max(1e-12,bef)),
              'rigid_same_geometry_heldout':rigid,
              'shape_parameter_mae':float(np.mean(np.abs(np.r_[controls,lobes]-np.r_[fit['shape'].controls,fit['shape'].lobes])))}
        out.append(item);print('CASE',i,item,flush=True)
    summary={'mean_gain_percent':float(np.mean([x['heldout_gain_percent'] for x in out])),
        'mean_shape_parameter_mae':float(np.mean([x['shape_parameter_mae'] for x in out])),
        'mean_unseen_error_before':float(np.mean([x['unseen_signal_before'] for x in out])),
        'mean_unseen_error_after':float(np.mean([x['unseen_signal_after'] for x in out]))}
    report={'version':'HDR WaveForm v0.4', 'claim':'linearized bidirectional 3D acoustic pressure and XYZ soft membrane',
     'limitations':['same numerical model produces both synthetic reference and reconstructed responses',
     'no experimental voice data', 'fixed voxel fluid mask and small deformation approximation',
     'no automatic topology change or global uniqueness proof',
     'no comparative evidence for general intelligence'],
     'config':CONFIG,'cases':out,'summary':summary,'runtime_seconds':round(time.perf_counter()-started,2)}
    p=Path(__file__).resolve().parents[1]/'results/benchmark.json';p.write_text(json.dumps(report,indent=2))
    print('SUMMARY',summary,flush=True)

if __name__=='__main__':main()
