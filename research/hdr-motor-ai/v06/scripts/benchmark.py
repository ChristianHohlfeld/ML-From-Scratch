"""Independent-input generalization gates: train/val/test and unseen combinations.

Never use test or withheld-quadrant samples to select a mechanical topology.
This measures a software simulator on SYNTHETIC data, not analog hardware.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from gear_ai import Stage,Machine,learn,loss


def truth(task,a,b):
    if task=='compound':
        M=Machine(.05+.08j,[Stage(1,0,.23,.16),Stage(0,-2,.14,-.2),Stage(2,-1,.42,.09),
                              Stage(-1,2,.20,.15),Stage(1,1,.16,-.08)])
        return M.complex_sample(a,b)
    if task=='additive':
        M=Machine(-.07+.04j,[Stage(1,0,.23,.16),Stage(-2,0,.1,-.2),Stage(0,1,.38,.1),Stage(0,-2,.12,.09)])
        return M.complex_sample(a,b)
    if task=='sharp':
        # 2D nonlinear interactions; not a sparse series of compound rotations.
        v=np.tanh(4*np.sin(a)*np.sin(b))
        u=np.tanh(5*np.cos(a-b))
        return (.3*v+.08*np.sin(2*a)) + 1j*(.28*u+.09*np.cos(b))
    raise ValueError(task)


def mlp_fit(a,b,y,ta,tb,ty,seed,epochs=450,width=32):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    X=lambda a,b:torch.tensor(np.stack((np.sin(a),np.cos(a),np.sin(b),np.cos(b)),axis=-1),dtype=torch.float32)
    Y=lambda z:torch.tensor(np.stack((z.real,z.imag),axis=-1),dtype=torch.float32)
    net=torch.nn.Sequential(torch.nn.Linear(4,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,2))
    opt=torch.optim.Adam(net.parameters(),lr=.015)
    x=X(a,b);t=Y(y)
    # choose a deterministic fixed training budget, no test/validation feedback
    t0=time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad();p=net(x);error=torch.nn.functional.mse_loss(p,t);error.backward();opt.step()
    dur=time.perf_counter()-t0
    with torch.no_grad():
        p=net(X(ta,tb)).numpy()
    return float(np.mean(abs((p[:,0]+1j*p[:,1])-ty)**2)),dur,sum(q.numel() for q in net.parameters())


def split(seed):
    rng=np.random.default_rng(seed)
    a=rng.uniform(-math.pi,math.pi,10000);b=rng.uniform(-math.pi,math.pi,10000)
    hold=(a>0)&(b>0)
    eligible=np.flatnonzero(~hold)
    withheld=np.flatnonzero(hold)
    rng.shuffle(eligible);rng.shuffle(withheld)
    tr=eligible[:520];val=eligible[520:690];test=eligible[690:1110];ood=withheld[:420]
    assert not np.intersect1d(tr,ood).size
    return rng,a,b,tr,val,test,ood


def run(task,seed,epochs):
    rng,a,b,tr,val,test,ood=split(seed)
    target=truth(task,a,b)
    sigma=.035
    noisy=target + rng.normal(0,sigma,len(a))+1j*rng.normal(0,sigma,len(a))
    train=(a[tr],b[tr]);validation=(a[val],b[val]);iid=(a[test],b[test]);extra=(a[ood],b[ood])
    model={}
    for typ,mixed in [('compound',True),('additive_only',False)]:
        t0=time.perf_counter()
        r=learn(train,noisy[tr],validation,noisy[val],max_ratio=3,max_stages=8,penalty=.00025,mixed=mixed)
        model[typ]={'iid_mse':loss(r.machine,iid,target[test]),'new_combinations_mse':loss(r.machine,extra,target[ood]),
                    'stages':[[q.k,q.l] for q in r.machine.stages],'parameters':2+2*len(r.machine.stages),
                    'fit_seconds':time.perf_counter()-t0,'events':r.steps}
    mlpiid,sec,pars=mlp_fit(*train,noisy[tr],*iid,target[test],seed,epochs)
    mlpood,_,_=mlp_fit(*train,noisy[tr],*extra,target[ood],seed,epochs)
    model['mlp']={'iid_mse':mlpiid,'new_combinations_mse':mlpood,'parameters':pars,'fit_seconds_one_run':sec}
    return {'task':task,'seed':seed,'n_train':len(tr),'n_validation':len(val),'n_test_iid':len(test),
            'n_test_new_combinations':len(ood),'noise_std_per_channel':sigma,'models':model}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',default='3,7,11');ap.add_argument('--out',default='results/benchmark.json');ap.add_argument('--epochs',type=int,default=350)
    opt=ap.parse_args()
    results=[]
    for task in ['compound','additive','sharp']:
        for seed in [int(s) for s in opt.seeds.split(',')]:
            r=run(task,seed,opt.epochs);results.append(r)
            print(task,seed,[(k,round(v['iid_mse'],5),round(v['new_combinations_mse'],5)) for k,v in r['models'].items()],flush=True)
    p=Path(opt.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'epochs':opt.epochs,'results':results},indent=2))
    print('saved',p)
