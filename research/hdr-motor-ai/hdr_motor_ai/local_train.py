"""Reproducible equilibrium-propagation and baseline comparison runner."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .mechanics import GearNetwork
from .local_learning import local_update, predict_local, scalar_hub_accuracy
from .tasks import make_dataset, TASKS
from .train import TinyMLP


def run(task='group_contrast', *, seed=7, epochs=12, train_samples=256,
        validation_samples=256, test_samples=512, batch_size=64,
        free_steps=260, nudged_steps=260, beta=0.15, lr=0.15,
        output='results/v04/ep', baseline_epochs=16,
        learn=True, threads=2):
    if min(epochs,train_samples,validation_samples,test_samples,batch_size,free_steps,nudged_steps) < 1:
        raise ValueError('All loop lengths must be positive')
    torch.set_num_threads(threads)
    random.seed(seed); torch.manual_seed(seed)
    xtr,ytr=make_dataset(train_samples,seed+11,task)
    xval,yval=make_dataset(validation_samples,seed+137,task)
    xte,yte=make_dataset(test_samples,seed+539,task)
    model=GearNetwork(channels=1)
    # No motor biases / motor gains are trained; no readout weight used.
    start=time.monotonic(); best_val=-1.; best_state=None; history=[]
    for epoch in range(epochs):
        perm=torch.randperm(len(xtr))
        for j in range(0,len(perm),batch_size):
            ix=perm[j:j+batch_size]
            local_update(model,xtr[ix],ytr[ix],free_steps=free_steps,nudged_steps=nudged_steps,
                         beta=beta,lr=lr,fit=learn)
        val=scalar_hub_accuracy(predict_local(model,xval,free_steps),yval)
        history.append({'epoch':epoch+1,'validation_accuracy':val})
        if val > best_val:
            best_val=val
            best_state={k:v.clone() for k,v in model.state_dict().items()}
        print(f'EP epoch {epoch+1}/{epochs}: val={val:.3f}',flush=True)
    model.load_state_dict(best_state)
    ep_test=scalar_hub_accuracy(predict_local(model,xte,free_steps),yte)
    ep_seconds=time.monotonic()-start
    # Comparable conventional baseline, same data splits, validation-selected.
    torch.manual_seed(seed)
    mlp=TinyMLP()
    optim=torch.optim.AdamW(mlp.parameters(),lr=0.015)
    baseline_best=float('inf'); baseline_state=None
    t0=time.monotonic()
    for epoch in range(baseline_epochs):
        perm=torch.randperm(len(xtr))
        for j in range(0,len(perm),batch_size):
            ix=perm[j:j+batch_size]
            optim.zero_grad(); logits=mlp(xtr[ix]); loss=F.binary_cross_entropy_with_logits(logits,ytr[ix]);loss.backward();optim.step()
        with torch.no_grad():val_loss=float(F.binary_cross_entropy_with_logits(mlp(xval),yval))
        if val_loss<baseline_best:
            baseline_best=val_loss; baseline_state={k:v.detach().clone() for k,v in mlp.state_dict().items()}
    mlp.load_state_dict(baseline_state)
    with torch.no_grad():mlp_acc=float(((mlp(xte)>=0).float()==yte).float().mean())
    result={'version':'0.4.0','task':task,'seed':seed,'train_examples':train_samples,'val_examples':validation_samples,
            'test_examples':test_samples,'steps_per_phase':[free_steps,nudged_steps],
            'beta':beta,'lr':lr,'local_learning':learn,'epochs':epochs,'baseline_epochs':baseline_epochs,
            'ep_best_validation_accuracy':best_val,'ep_test_accuracy':ep_test,'mlp_test_accuracy':mlp_acc,
            'ep_train_seconds':round(ep_seconds,3),'mlp_train_seconds':round(time.monotonic()-t0,3),
            'method':'Two-phase local equilibrium propagation; supervised V0 teacher torque during nudged phase only; NO backpropagation for gear learning',
            'limitations':'Simulated on CPU, not physical hardware; finite settling and discrete time can bias local EP gradients.'}
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    (path/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (path/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    (path/'learned_gears.json').write_text(json.dumps(model.browser_export(),indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS,default='group_contrast');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--baseline-epochs',type=int,default=16)
    p.add_argument('--train-samples',type=int,default=256);p.add_argument('--val-samples',type=int,default=256)
    p.add_argument('--test-samples',type=int,default=512);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--free-steps',type=int,default=260);p.add_argument('--nudged-steps',type=int,default=260)
    p.add_argument('--lr',type=float,default=0.15);p.add_argument('--beta',type=float,default=0.15)
    p.add_argument('--output',default='results/v04/ep');p.add_argument('--no-learning',action='store_true')
    a=p.parse_args()
    run(a.task,seed=a.seed,epochs=a.epochs,train_samples=a.train_samples,
        validation_samples=a.val_samples,test_samples=a.test_samples,batch_size=a.batch_size,
        free_steps=a.free_steps,nudged_steps=a.nudged_steps,lr=a.lr,beta=a.beta,
        output=a.output,learn=not a.no_learning,baseline_epochs=a.baseline_epochs)
if __name__=='__main__':main()
