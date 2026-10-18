"""Reproducible, leakage-safe v0.5 study: validation selects structure; test is unseen."""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
from gear_ai import Gear, Design, search, fit_structure, mse


def dataset(task, x):
    p=2*np.pi*x
    if task=='mechanical':
        return Design(.14,-.09,[Gear(1,.73,.13),Gear(-2,.24,.19),Gear(4,-.22,.07),Gear(7,.09,-.11)]).sample(p)
    if task=='nonharmonic':
        # Cusps are deliberately difficult for sparse, low-order gears.
        return np.column_stack((np.sign(np.sin(p))*np.abs(np.sin(p))**.33,
                                np.sign(np.cos(p))*np.abs(np.cos(p))**.33))
    raise ValueError(task)


def run(task='mechanical',seed=3,ntrain=80,nval=80,ntest=200,max_gears=9,max_ratio=9,
        penalty=.00004,mlp_epochs=600,noise=.035):
    rng=np.random.default_rng(seed)
    xtr=rng.uniform(0,1,ntrain);xv=rng.uniform(0,1,nval)
    # Unseen test phases, fixed offset grid, not shared with train or validation.
    xte=(np.arange(ntest)+.35)/ntest
    ytr=dataset(task,xtr)+rng.normal(scale=noise,size=(ntrain,2))
    yv=dataset(task,xv)+rng.normal(scale=noise,size=(nval,2))
    yte=dataset(task,xte)
    t=time.perf_counter()
    selected=search(2*np.pi*xtr,ytr,2*np.pi*xv,yv,max_ratio=max_ratio,max_gears=max_gears,penalty=penalty)
    tgear=time.perf_counter()-t
    # Fixed-size dense Fourier baseline with same signed-gear dictionary, no pruning.
    dense=fit_structure(2*np.pi*xtr,ytr,[k for k in range(-max_ratio,max_ratio+1) if k],ridge=1e-5)
    # Exact least-squares results are a stronger comparison than a weak random baseline.
    res={'task':task,'seed':seed,'train_samples':ntrain,'val_samples':nval,'test_samples':ntest,
         'noise_sd':noise,'search_seconds':tgear,'selected_ratios':[g.turns for g in selected.design.gears],
         'selected_gears':len(selected.design.gears),'gear_test_mse':mse(selected.design,2*np.pi*xte,yte),
         'gear_val_mse':mse(selected.design,2*np.pi*xv,yv),
         'zero_gear_test_mse':mse(fit_structure(2*np.pi*xtr,ytr,[]),2*np.pi*xte,yte),
         'dense_fourier_test_mse':mse(dense,2*np.pi*xte,yte),
         'dense_gears':len(dense.gears),'steps':selected.history,
         'design':selected.design.to_json()}
    try:
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2))
        opt=torch.optim.Adam(model.parameters(),lr=.015)
        a=torch.tensor(np.column_stack((np.sin(2*np.pi*xtr),np.cos(2*np.pi*xtr))),dtype=torch.float32)
        b=torch.tensor(ytr,dtype=torch.float32)
        val_a=torch.tensor(np.column_stack((np.sin(2*np.pi*xv),np.cos(2*np.pi*xv))),dtype=torch.float32)
        val_b=torch.tensor(yv,dtype=torch.float32)
        te_a=torch.tensor(np.column_stack((np.sin(2*np.pi*xte),np.cos(2*np.pi*xte))),dtype=torch.float32)
        t=time.perf_counter();best=float('inf');state=None;pat=0
        for epoch in range(mlp_epochs):
            model.train();opt.zero_grad(set_to_none=True)
            loss=((model(a)-b)**2).mean();loss.backward();opt.step()
            if epoch%10==0:
                model.eval()
                with torch.no_grad(): vl=float(((model(val_a)-val_b)**2).mean())
                if vl<best-1e-8:
                    best=vl;state={k:v.detach().clone() for k,v in model.state_dict().items()};pat=0
                else:pat+=1
                if pat>=25:break
        model.load_state_dict(state)
        with torch.no_grad(): res['mlp_test_mse']=float(((model(te_a)-torch.tensor(yte,dtype=torch.float32))**2).mean())
        res['mlp_parameters']=sum(p.numel() for p in model.parameters())
        res['mlp_train_seconds']=time.perf_counter()-t
        res['mlp_epochs']=epoch+1
    except ImportError:
        res['mlp_test_mse']=None
        res['mlp_note']='PyTorch unavailable, baseline skipped'
    return res

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results/v05')
    parser.add_argument('--seeds',default='3,7,11')
    parser.add_argument('--epochs',type=int,default=600)
    args=parser.parse_args()
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in ['mechanical','nonharmonic']:
        for seed in map(int,args.seeds.split(',')):
            r=run(task,seed,mlp_epochs=args.epochs)
            dest.joinpath(f'{task}_{seed}.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
            rows.append({k:v for k,v in r.items() if isinstance(v,(float,int,str)) or v is None})
            print(f"{task}/{seed}: {r['selected_gears']} gears; test MSE {r['gear_test_mse']:.6g}, "
                  f"dense {r['dense_fourier_test_mse']:.6g}, MLP {r['mlp_test_mse']:.6g}")
    with open(dest/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
