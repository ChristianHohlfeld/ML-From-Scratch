"""Honest inverse-acoustic shape learning using the 3D wave simulator.

Derivative-free / black-box updates change the *geometry*, not the wave
physics. Independent excitation is used to assess generalization.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from .core import Shape, propagate, signature


def fit_shape(target, *, initial=None, max_evals=72, steps=2100,
              excitation='chirp', nx=38, ny=19, nz=19):
    """Optimize 4 shape controls. True topology, length, aspect are fixed.

    Cannot infer all anatomical structure or distinguish degenerate forms.
    Only known normalized four-control radius family is considered.
    """
    if initial is None:initial=Shape()
    shape=initial
    cache={}
    best={'loss':float('inf'),'controls':shape.controls,'evals':0}
    def loss(v):
        v=tuple(float(x) for x in np.clip(v,.5,1.48))
        key=tuple(round(x,5) for x in v)
        if key not in cache:
            s=replace(shape,controls=v)
            pred=signature(propagate(s,steps=steps,excitation=excitation,nx=nx,ny=ny,nz=nz))
            cache[key]=float(np.mean((pred-target)**2))
            best['evals']+=1
            if cache[key]<best['loss']:
                best.update(loss=cache[key],controls=v)
        return cache[key]
    before=loss(np.asarray(shape.controls))
    result=minimize(loss,np.asarray(shape.controls),method='Powell',
                    bounds=[(.5,1.48)]*4,
                    options={'maxfev':max_evals,'ftol':1e-4,'xtol':.025})
    return {'before':before,'after':best['loss'],
            'shape':replace(shape,controls=tuple(best['controls'])),
            'evals':best['evals'],'optimized':bool(result.success),
            'message':str(result.message)}
