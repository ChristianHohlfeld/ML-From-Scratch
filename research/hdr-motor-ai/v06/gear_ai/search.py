"""Offline sparse structural search over physically implementable compound gears.

A term (k,l) rotates by k*shaftA+l*shaftB. Searching the sparse dictionary
is a conventional OMP-like statistical optimization, NOT physical learning.
At inference every selected gear is fixed and only the shafts move.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .core import Stage,Machine


def dictionary(max_ratio=4,mixed=True):
    if max_ratio<1:raise ValueError('max_ratio must be positive')
    return [(k,l) for k in range(-max_ratio,max_ratio+1)
            for l in range(-max_ratio,max_ratio+1) if (k or l) and (mixed or not(k and l))]


def _phi(inputs,ks):
    a,b=map(lambda x:np.asarray(x,dtype=float).reshape(-1),inputs)
    if a.shape!=b.shape:raise ValueError('Input phases mismatch')
    return np.column_stack([np.ones(len(a),complex)]+[np.exp(1j*(k*a+l*b)) for k,l in ks])


def fit(inputs,target,ks,ridge=1e-7):
    z=np.asarray(target,dtype=complex).reshape(-1)
    if z.size!=np.asarray(inputs[0]).size:raise ValueError('Target size mismatch')
    if not (np.isfinite(z).all() and np.isfinite(inputs).all()):raise ValueError('Nonfinite data')
    if len(ks)!=len(set(ks)) or any(not(k or l) for k,l in ks):raise ValueError('Invalid structure')
    A=_phi(inputs,ks)
    reg=np.eye(A.shape[1]);reg[0,0]=0
    c=np.linalg.solve(A.conj().T@A+ridge*reg+1e-10*np.eye(A.shape[1]),A.conj().T@z)
    return Machine(c[0],[Stage(k,l,c[j+1].real,c[j+1].imag) for j,(k,l) in enumerate(ks)])


def loss(model,inputs,z):
    return float(np.mean(abs(model.complex_sample(*inputs)-z)**2))

@dataclass
class Result:
    machine: Machine
    steps: list
    initial_loss: float
    final_val_loss: float


def learn(train_inputs,train_y,val_inputs,val_y,max_ratio=4,max_stages=8,penalty=4e-5,mixed=True):
    """Greedy add and prune; validation selects, TEST samples are never used."""
    if max_stages<0 or penalty<0:raise ValueError('Invalid budget')
    pool=dictionary(max_ratio=max_ratio,mixed=mixed) # independent of the input sample
    ks=[]
    def score(ks):
        model=fit(train_inputs,train_y,ks)
        e=loss(model,val_inputs,val_y)
        return e+penalty*len(ks),e,model
    best,err,machine=score(ks)
    initial=best
    trace=[dict(event='initial',gears=0,score=best,val_error=err)]
    for _ in range(max_stages):
        # Screening by measured residual frequency avoids fitting all candidate systems.
        residual=np.asarray(train_y)-machine.complex_sample(*train_inputs)
        corr=[]
        for k,l in pool:
            if (k,l) in ks:continue
            phase=np.exp(-1j*(k*np.asarray(train_inputs[0])+l*np.asarray(train_inputs[1])))
            corr.append((abs(np.mean(residual*phase)),k,l))
        if not corr:break
        corr.sort(reverse=True)
        options=[]
        for _,k,l in corr[:min(12,len(corr))]:
            sc,e,m=score(ks+[(k,l)])
            options.append((sc,k,l,e,m))
        sc,k,l,e,m=min(options,key=lambda v:(v[0],abs(v[1])+abs(v[2]),v[1],v[2]))
        if sc>=best-1e-10:break
        ks.append((k,l));best,err,machine=sc,e,m
        trace.append(dict(event='add',gears=len(ks),k=k,l=l,score=sc,val_error=e))
    changed=True
    while changed:
        changed=False
        for k,l in sorted(ks,key=lambda t:next(g.radius for g in machine.stages if (g.k,g.l)==t)):
            p=[t for t in ks if t!=(k,l)]
            sc,e,m=score(p)
            if sc<best-1e-10:
                ks=p;best,err,machine=sc,e,m;changed=True
                trace.append(dict(event='prune',gears=len(ks),k=k,l=l,score=sc,val_error=e))
                break
    return Result(machine,trace,initial,err)
