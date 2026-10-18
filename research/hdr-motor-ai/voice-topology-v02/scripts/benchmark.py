"""Reproducible inverse geometry benchmark (synthetic, matched simulator)."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from waveform import Shape, propagate,signature,fit_shape

cases=[('A wide-narrow',(.69,1.25,.72,1.3)),
       ('B subtle',(.95,1.14,.90,1.05)),
       ('C alternating',(1.32,.66,1.30,.72))]
baseline=Shape((1.,1.,1.,1.));steps=2100;out=[]
for name,controls in cases:
    truth=Shape(controls)
    target_train=signature(propagate(truth,steps=steps,excitation='chirp'))
    target_test=signature(propagate(truth,steps=steps,excitation='multitone'))
    start=time.perf_counter()
    learned=fit_shape(target_train,initial=baseline,max_evals=140,steps=steps)
    b_train=float(np.mean((signature(propagate(baseline,steps=steps,excitation='chirp'))-target_train)**2))
    a_train=float(np.mean((signature(propagate(learned['shape'],steps=steps,excitation='chirp'))-target_train)**2))
    b_test=float(np.linalg.norm(signature(propagate(baseline,steps=steps,excitation='multitone'))-target_test))
    a_test=float(np.linalg.norm(signature(propagate(learned['shape'],steps=steps,excitation='multitone'))-target_test))
    q={'case':name,'truth':controls,'recovered':learned['shape'].controls,
       'baseline_train_mse':b_train,'learned_train_mse':a_train,
       'baseline_test_l2':b_test,'learned_test_l2':a_test,
       'train_evaluations':learned['evals'],'seconds':round(time.perf_counter()-start,3),
       'holdout_improvement_pct':float(100*(1-a_test/b_test))}
    out.append(q);print(json.dumps(q),flush=True)
summary={'protocol':'Full 3D FDTD, 4 variable radii, fixed 3D tube topology/length/ellipse aspect; train chirp, held-out multitone; comparison to untrained flat 3D tube. Entire study uses synthetic outputs of the same discretized simulator.','num_cases':len(out),'cases':out,'mean_holdout_improvement_pct':float(np.mean([x['holdout_improvement_pct'] for x in out])), 'mean_target_geometry_parameter_absolute_error':float(np.mean([np.mean(np.abs(np.asarray(x['truth'])-np.asarray(x['recovered']))) for x in out])), 'notes':['Tuned inverse problem, NOT measured human anatomy','Distinct shapes may be acoustically indistinguishable with one sensor','Cross-sectional H1 is supplied by the shape parameter, not inferred from acoustic data','FDTD walls reflect; quasi-static shape updates, not a fully two-way deforming soft-body','Dimensionless pressure amplitude; no hardware calibration or energy-efficiency claims']}
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/benchmark.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps({'mean_holdout_improvement_pct':summary['mean_holdout_improvement_pct'],'mean_shape_parameter_abs_error':summary['mean_target_geometry_parameter_absolute_error']},indent=2))
