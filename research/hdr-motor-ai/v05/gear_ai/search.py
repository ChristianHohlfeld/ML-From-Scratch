"""Offline structural learning by residual response; no global solver at inference.

Machine is a signed harmonic gear-chain. Structure search tests add/remove of
integer ratios, refits amplitudes/phases, scores on validation and penalizes parts.
This is a sparse Fourier model/search, NOT novel universal learning.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import Gear, Design


def _system(phases, ks):
    p=np.asarray(phases,dtype=float).reshape(-1)
    A=np.zeros((2*len(p), 2+2*len(ks)))
    A[0::2,0]=1.; A[1::2,1]=1.
    for j,k in enumerate(ks):
        c=np.cos(k*p);s=np.sin(k*p)
        A[0::2,2+2*j]=c
        A[0::2,3+2*j]=-s
        A[1::2,2+2*j]=s
        A[1::2,3+2*j]=c
    return A


def fit_structure(phases, xy, ks, ridge=1.e-8):
    """Least-squares calibration only when topology changes (training time)."""
    phases=np.asarray(phases,dtype=float)
    xy=np.asarray(xy,dtype=float)
    if xy.shape!=(len(phases),2): raise ValueError('Need N x 2 measured positions')
    if not np.isfinite(xy).all() or not np.isfinite(phases).all(): raise ValueError('Finite inputs required')
    if len(set(ks)) != len(ks) or 0 in ks: raise ValueError('Unique nonzero signed ratios required')
    A=_system(phases,ks)
    target=xy.reshape(-1)
    reg=np.eye(A.shape[1]);reg[0,0]=reg[1,1]=0
    params=np.linalg.solve(A.T@A+ridge*reg+1e-12*np.eye(A.shape[1]),A.T@target)
    return Design(float(params[0]),float(params[1]),[Gear(int(k),float(params[2+2*j]),float(params[3+2*j])) for j,k in enumerate(ks)])


def mse(design,phases,xy):
    return float(np.mean((design.sample(phases)-np.asarray(xy))**2))

@dataclass
class SearchResult:
    design: Design
    history: list
    initial_score: float


def search(phases_train, xy_train, phases_val, xy_val, max_ratio=10, max_gears=10,
           penalty=1.e-5, min_gain=1.e-10, ridge=1.e-7):
    """Greedy add, refit, prune. Held-out evaluation must use OTHER data.

    Only integer gear ratios allowed. Penalty counts real mechanical stages.
    The validation set is used for architecture selection, not final reporting.
    """
    if max_ratio<1 or max_gears<0 or penalty<0: raise ValueError('Invalid search limits')
    pool=[k for k in range(-max_ratio,max_ratio+1) if k]
    ks=[]
    def train(ks):
        d=fit_structure(phases_train,xy_train,ks,ridge)
        val=mse(d,phases_val,xy_val)
        return d,val,val+penalty*len(ks)
    best,bv,bs=train(ks)
    initial=bs
    history=[{'event':'initial','gears':0,'val_mse':bv,'score':bs,'k':None}]
    for step in range(max_gears):
        options=[]
        for k in pool:
            if k in ks:continue
            d,v,s=train(ks+[k]);options.append((s,k,d,v))
        if not options:break
        s,k,d,v=min(options,key=lambda t:(t[0],abs(t[1]),t[1]))
        if not s < bs-min_gain:break
        ks.append(k);best,bv,bs=d,v,s
        history.append({'event':'add','gears':len(ks),'val_mse':bv,'score':bs,'k':k})
    changed=True
    while changed and ks:
        changed=False
        for k in sorted(ks,key=lambda k: next(g.length for g in best.gears if g.turns==k)):
            sub=[q for q in ks if q!=k]
            d,v,s=train(sub)
            if s<bs-min_gain:
                ks=sub;best,bv,bs=d,v,s;changed=True
                history.append({'event':'prune','gears':len(ks),'val_mse':bv,'score':bs,'k':k})
                break
    return SearchResult(best,history,initial)
