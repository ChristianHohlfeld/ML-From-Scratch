"""3D phase-space shape learning for voiced acoustic signals.

A phase-space embedding of single-channel sound is NOT a physical 3D field and
cannot by itself identify the anatomical topology of the vocal tract.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, stft
from scipy.fft import dct
from scipy.spatial.distance import pdist, squareform
from math import gcd

SR=8000
FORMANTS={'a':(800,1180,2700),'e':(520,1950,2700),'i':(290,2290,3010),
          'o':(480,920,2600),'u':(325,760,2400)}
BANDS=((100,350),(350,650),(650,1050),(1050,1600),(1600,2400),(2400,3800))
SCALES=(.22,.32,.44,.58,.76,1.)

def synth_vowel(vowel='a', f0=130, seed=0, noise=.004, seconds=.5):
    """A reproducible synthetic voiced source-filter vowel, NOT human speech."""
    rng=np.random.default_rng(seed)
    n=int(seconds*SR); t=np.arange(n)/SR
    phase=2*np.pi*f0*t+.022*np.sin(2*np.pi*3.3*t+seed)
    out=np.zeros(n)
    f=FORMANTS[vowel]
    for h in range(1,int(3800/f0)+1):
        hz=h*f0
        resonances=.022+sum(w*np.exp(-.5*((hz-c)/b)**2) for c,b,w in zip(f,(90,135,190),(1,.75,.38)))
        out+=(resonances/h**1.12)*np.sin(h*(phase+.003*rng.normal()))
    out+=rng.normal(0,noise,n)
    r=int(.035*SR)
    window=np.ones(n);window[:r]=np.sin(np.linspace(0,np.pi/2,r))**2;window[-r:]=window[:r][::-1]
    out*=window
    return (out/(np.max(np.abs(out))+1e-9)*.75).astype('float32')

def prepare(wave,sr=SR):
    x=np.asarray(wave,dtype=float)
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=SR: d=gcd(int(sr),SR);x=resample_poly(x,SR//d,int(sr)//d)
    if len(x)<512:raise ValueError('Recording too short')
    if not np.isfinite(x).all():raise ValueError('Invalid audio')
    x=x-x.mean()
    if np.max(np.abs(x))<1e-5:raise ValueError('Silent audio')
    # No additional complex voice activity detector; exclude leading/trailing silence.
    energy=np.sqrt(np.convolve(x*x,np.ones(160)/160,mode='same'))
    indices=np.flatnonzero(energy>max(.002,energy.max()*.12))
    if len(indices)<256:raise ValueError('Insufficient voiced sound')
    x=x[max(0,indices[0]-80):min(len(x),indices[-1]+81)]
    return x/(np.sqrt(np.mean(x*x))+1e-9)

def pitch_estimate(x):
    n=min(len(x),4096);x=x[(len(x)-n)//2:(len(x)+n)//2]-x.mean()
    c=np.correlate(x,x,'full')[n-1:]
    c=c/(np.arange(len(x),0,-1)+1e-9)
    lo,hi=SR//270,SR//75
    peaks=[p for p in range(lo+1,hi) if c[p]>=c[p-1] and c[p]>=c[p+1]]
    if not peaks:peaks=[lo+int(np.argmax(c[lo:hi]))]
    peak=max(c[p] for p in peaks)
    return SR/min(p for p in peaks if c[p]>=peak*.88)

def embed(x,lag,points=52):
    usable=len(x)-2*lag
    if usable<points:raise ValueError('Recording too short for 3D reconstruction')
    idx=np.linspace(int(.08*usable),int(.92*usable)-1,points).astype(int)
    cloud=np.column_stack([x[idx],x[idx+lag],x[idx+2*lag]])
    cloud-=cloud.mean(0)
    return cloud/(np.sqrt(np.mean(np.sum(cloud*cloud,axis=1)))+1e-9)

def betti_curve(points,scales=SCALES):
    """Finite-scale Vietoris-Rips clique-complex b0,b1, GF(2) triangle rank."""
    p=np.asarray(points);n=len(p)
    if n<3 or n>60:raise ValueError('Expected 3..60 landmarks')
    d=squareform(pdist(p));nz=d[d>1e-9];d/=np.median(nz) if len(nz) else 1.
    out=[]
    for r in scales:
        adjacency=d<=r;np.fill_diagonal(adjacency,False)
        edges=[];par=list(range(n))
        def find(i):
            while par[i]!=i:par[i]=par[par[i]];i=par[i]
            return i
        for i in range(n):
            for j in range(i+1,n):
                if adjacency[i,j]:
                    edges.append((i,j));a,b=find(i),find(j)
                    if a!=b:par[a]=b
        b0=len({find(i) for i in range(n)}); idx={p:k for k,p in enumerate(edges)};basis={}
        for i in range(n):
            for j in range(i+1,n):
                if not adjacency[i,j]:continue
                for k in range(j+1,n):
                    if not (adjacency[i,k] and adjacency[j,k]):continue
                    vec=(1<<idx[i,j])^(1<<idx[i,k])^(1<<idx[j,k])
                    while vec:
                        pivot=vec.bit_length()-1
                        if pivot not in basis:basis[pivot]=vec;break
                        vec^=basis[pivot]
        b1=max(0,len(edges)-n+b0-len(basis))
        out.extend((b0/n,b1/n))
    return np.array(out)

def signature(wave,sr=SR):
    """Multiscale wave ensemble: 3D geometry + loop counts + band coupling."""
    x=prepare(wave,sr);pitch=pitch_estimate(x)
    lag=int(np.clip(round(SR/(max(75,pitch)*4)),6,27))
    channels=[sosfiltfilt(butter(3,[a,b],btype='bandpass',fs=SR,output='sos'),x) for a,b in BANDS]
    en=np.array([np.sqrt(np.mean(y*y)) for y in channels]);en/=en.sum()+1e-9
    geo=[];top=[]
    for y in [x,*channels]:
        p=embed(y,lag)
        distances=np.sort(pdist(p))
        geo.extend(np.quantile(distances,np.linspace(.05,.95,10)))
        top.extend(betti_curve(p))
    env=np.array([np.sqrt(np.convolve(y*y,np.ones(48)/48,mode='same')) for y in channels])
    corr=np.nan_to_num(np.corrcoef(env),nan=0,posinf=0,neginf=0)
    couple=corr[np.triu_indices(6,1)]
    geo=np.asarray(geo);top=np.asarray(top)
    return {'pitch':pitch,'lag':lag,'cloud':embed(x,lag),'energy':en,'geometry':geo,
            'topology':top,'coupling':couple,'combined':np.r_[en*3,geo*.35,top*.18,couple*.16]}

def mfcc(wave,sr=SR):
    x=prepare(wave,sr)
    f,_,z=stft(x,SR,nperseg=256,noverlap=128,boundary=None)
    mel=lambda h:2595*np.log10(1+h/700)
    im=lambda m:700*(10**(m/2595)-1)
    cut=im(np.linspace(mel(80),mel(3900),28))
    bank=[]
    for lo,mid,hi in zip(cut[:-2],cut[1:-1],cut[2:]):
        tri=np.maximum(0,np.minimum((f-lo)/(mid-lo),(hi-f)/(hi-mid)))
        bank.append(np.maximum(1e-10,tri@np.abs(z)**2))
    co=dct(np.log(bank),axis=0,norm='ortho')[:13]
    return np.r_[co.mean(axis=1),co.std(axis=1)]
