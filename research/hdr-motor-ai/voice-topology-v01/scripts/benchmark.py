"""Synthetic-vowel held-out pitch generalization; includes honest spectral baseline."""
from pathlib import Path
import json,sys,time
import numpy as np
from hdr_voice import synth_vowel,signature,mfcc,FORMANTS

def nearest(train,labels,test):
    a=np.asarray(train);b=np.asarray(test)
    std=np.std(a,axis=0);nz=std[std>1e-8]
    std=np.maximum(std,np.percentile(nz,40) if len(nz) else 1)
    d=np.sum(((b[:,None,:]-a[None,:,:])/std)**2,axis=2)
    return [labels[i] for i in np.argmin(d,axis=1)]

def evaluate(seed=12,train_n=4,test_n=4):
    rng=np.random.default_rng(seed);data=[]
    for label in FORMANTS:
        for k in range(train_n+test_n):
            train=k<train_n
            hz=rng.uniform(98,170) if train else rng.uniform(205,255)
            x=synth_vowel(label,hz,seed=seed*100+7*k+list(FORMANTS).index(label))
            s=signature(x)
            data.append({'label':label,'train':train,'geometry':s['geometry'],'topology':s['topology'],'energy':s['energy'],'coupled':s['combined'],'mfcc':mfcc(x)})
    train=[x for x in data if x['train']];test=[x for x in data if not x['train']]
    answer=[x['label'] for x in test];results={}
    for feat in ('geometry','topology','energy','coupled','mfcc'):
        pred=nearest([x[feat] for x in train],[x['label'] for x in train],[x[feat] for x in test])
        results[feat]=sum(x==y for x,y in zip(pred,answer))/len(answer)
    return results

def main():
    start=time.monotonic();runs={str(seed):evaluate(seed) for seed in (12,19,31)}
    names=list(next(iter(runs.values())))
    summary={k:{'mean':float(np.mean([runs[s][k] for s in runs])),
                'per_seed':[runs[s][k] for s in runs]} for k in names}
    report={'dataset':'Toy synthetic voiced source-filter vowels a,e,i,o,u',
            'split':'Train pitches 98..170 Hz; test pitches 205..255 Hz; independent randomized phases/noise',
            'test_samples_per_seed':20,'baseline':'Nearest neighbour, per-feature standardized distance; 26-value MFCC baseline',
            'limitation':'Not real speech, 3D anatomy, an LLM, or evidence of superiority',
            'results':summary,'duration_seconds':round(time.monotonic()-start,2)}
    Path('results').mkdir(exist_ok=True)
    Path('results/benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
