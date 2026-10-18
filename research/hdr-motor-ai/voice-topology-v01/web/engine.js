/* HDR Voice Topology: acoustic phase-space reconstruction and shape learner.
 * Points are 3D DELAY COORDINATES, not physical 3D acoustic-space coordinates.
 * This script is standalone, self-hosted and has no network dependencies.
 */
(()=>{'use strict';
const SR=8000,FORMANTS={a:[800,1180,2700],e:[520,1950,2700],i:[290,2290,3010],o:[480,920,2600],u:[325,760,2400]};
const SOS=__SOS__;
const app={active:0,slots:[null,null],memory:[],yaw:.25,pitch:-.25,zoom:1,stream:null,recorder:null};
const $=id=>document.getElementById(id),TAU=2*Math.PI;
function seeded(seed){let x=seed>>>0||1;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};}
function synth(v,f0,seed=2){const fs=FORMANTS[v],rand=seeded(seed),N=4000,s=new Float32Array(N),phase=Array.from({length:N},(_,j)=>TAU*f0*j/SR+.022*Math.sin(TAU*3.3*j/SR+seed));
 for(let h=1;h*f0<3800;h++){const hz=h*f0,b=[90,135,190],weight=[1,.75,.38];let response=.022;
  for(let k=0;k<3;k++)response+=weight[k]*Math.exp(-.5*((hz-fs[k])/b[k])**2);
  const scale=response/h**1.12,shift=(rand()-.5)*.02;
  for(let j=0;j<N;j++)s[j]+=scale*Math.sin(h*phase[j]+shift);
 }
 let m=0;for(let j=0;j<N;j++){s[j]+=(rand()-.5)*.001;m=Math.max(m,Math.abs(s[j]));}
 for(let j=0;j<N;j++){const edge=Math.min(1,j/280,(N-1-j)/280);s[j]*=.75/(m+1e-9)*Math.sin(Math.PI*.5*edge)**2;}
 return s;}
function prep(data,sr){const n=Math.floor(data.length*SR/sr),x=new Float64Array(n);for(let j=0;j<n;j++){const f=j*sr/SR,i=Math.floor(f),t=f-i;x[j]=(1-t)*data[Math.min(i,data.length-1)]+t*data[Math.min(i+1,data.length-1)];}
 if(n<512)throw Error('Audio too short');let mean=0,max=0;for(const y of x)mean+=y;mean/=n;
 for(let j=0;j<n;j++){x[j]-=mean;max=Math.max(max,Math.abs(x[j]));}if(max<1e-5)throw Error('Audio is silent');
 let sum=0,energy=new Float64Array(n),mx=0;
 for(let j=0;j<n;j++){sum+=x[j]**2;if(j>=160)sum-=x[j-160]**2;energy[j]=Math.sqrt(Math.max(0,sum/Math.min(j+1,160)));mx=Math.max(mx,energy[j]);}
 let first=0,last=n-1;while(first<last&&energy[first]<mx*.12)first++;while(last>first&&energy[last]<mx*.12)last--;
 const z=x.slice(Math.max(0,first-80),Math.min(n,last+81));if(z.length<256)throw Error('Not enough voiced sound');
 let rms=0;for(const v of z)rms+=v*v;rms=Math.sqrt(rms/z.length)+1e-9;for(let j=0;j<z.length;j++)z[j]/=rms;return z;}
function f0estimate(x){const n=Math.min(4096,x.length),start=Math.floor((x.length-n)/2),low=Math.floor(SR/270),hi=Math.floor(SR/75);let mean=0;
 for(let i=0;i<n;i++)mean+=x[start+i];mean/=n;
 const c=new Float64Array(hi+2);
 for(let lag=low;lag<=hi+1;lag++){let sum=0;for(let i=0;i<n-lag;i++)sum+=(x[start+i]-mean)*(x[start+i+lag]-mean);c[lag]=sum/(n-lag);}
 let peaks=[];for(let lag=low+1;lag<hi;lag++)if(c[lag]>=c[lag-1]&&c[lag]>=c[lag+1])peaks.push(lag);
 if(!peaks.length){let best=low;for(let j=low+1;j<hi;j++)if(c[j]>c[best])best=j;peaks=[best];}
 const max=Math.max(...peaks.map(j=>c[j]));return SR/peaks.find(j=>c[j]>=max*.88);}
function filterSOS(x,stages){function filt(s){let input=Float64Array.from(s);for(const [b0,b1,b2,a0,a1,a2] of stages){const out=new Float64Array(input.length);let z1=0,z2=0;for(let i=0;i<input.length;i++){const y=b0*input[i]+z1;z1=b1*input[i]-a1*y+z2;z2=b2*input[i]-a2*y;out[i]=y;}input=out;}return input;}
 return Float64Array.from(filt(Float64Array.from(filt(x)).reverse())).reverse();}
function embed(x,lag,points=52){const n=x.length-2*lag,cloud=[],mean=[0,0,0];if(n<points)throw Error('Recording too short');
 for(let i=0;i<points;i++){const j=Math.floor(n*.08+(n*.84-1)*i/(points-1)),p=[x[j],x[j+lag],x[j+2*lag]];cloud.push(p);for(let k=0;k<3;k++)mean[k]+=p[k]/points;}
 let varsum=0;for(const p of cloud){for(let k=0;k<3;k++)p[k]-=mean[k];varsum+=p[0]**2+p[1]**2+p[2]**2;}
 const r=Math.sqrt(varsum/points)||1;for(const p of cloud)for(let k=0;k<3;k++)p[k]/=r;return cloud;}
function distance(a,b){let v=0;for(let k=0;k<3;k++)v+=(a[k]-b[k])**2;return Math.sqrt(v);}
function quantile(sorted,p){let q=p*(sorted.length-1),i=Math.floor(q);return sorted[i]*(1-(q-i))+sorted[Math.min(i+1,sorted.length-1)]*(q-i);}
function distances(pts){let d=[];for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++)d.push(distance(pts[i],pts[j]));return d.sort((a,b)=>a-b);}
function betti(pts){const n=pts.length,sorted=distances(pts).filter(x=>x>1e-9),norm=quantile(sorted,.5)||1,dist=Array.from({length:n},()=>new Float64Array(n)),vals=[];
 for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)dist[i][j]=dist[j][i]=distance(pts[i],pts[j])/norm;
 for(const radius of [.22,.32,.44,.58,.76,1]){const parent=Array.from({length:n},(_,i)=>i),adj=Array.from({length:n},()=>new Uint8Array(n)),edges=[];
  const find=i=>{while(parent[i]!==i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(dist[i][j]<=radius){adj[i][j]=adj[j][i]=1;edges.push([i,j]);const a=find(i),b=find(j);if(a!==b)parent[a]=b;}
  const components=new Set(parent.map((_,i)=>find(i))).size,index=new Map(edges.map(([i,j],k)=>[i*n+j,k])),pivots=new Map();
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(adj[i][j])for(let k=j+1;k<n;k++)if(adj[i][k]&&adj[j][k]){
   let bits=(1n<<BigInt(index.get(i*n+j)))^(1n<<BigInt(index.get(i*n+k)))^(1n<<BigInt(index.get(j*n+k)));
   while(bits){const p=bits.toString(2).length-1;if(!pivots.has(p)){pivots.set(p,bits);break;}bits^=pivots.get(p);}
  }
  vals.push(components/n,Math.max(0,edges.length-n+components-pivots.size)/n);
 }
 return vals;}
function correlation(a,b){let ma=0,mb=0;for(let i=0;i<a.length;i++){ma+=a[i];mb+=b[i];}ma/=a.length;mb/=a.length;let num=0,va=0,vb=0;for(let i=0;i<a.length;i++){const x=a[i]-ma,y=b[i]-mb;num+=x*y;va+=x*x;vb+=y*y;}return num/Math.sqrt(Math.max(va*vb,1e-12));}
function envelope(x){let s=0,e=new Float64Array(x.length);for(let i=0;i<x.length;i++){s+=x[i]*x[i];if(i>=48)s-=x[i-48]*x[i-48];e[i]=Math.sqrt(Math.max(0,s/Math.min(i+1,48)));}return e;}
function signature(data,sr){const x=prep(data,sr),f0=f0estimate(x),lag=Math.max(6,Math.min(27,Math.round(SR/(Math.max(75,f0)*4))));
 const channels=SOS.map(s=>filterSOS(x,s)),wave=embed(x,lag),energy=channels.map(x=>Math.sqrt(x.reduce((sum,y)=>sum+y*y,0)/x.length)),tot=energy.reduce((a,b)=>a+b,0)||1;
 const en=energy.map(v=>v/tot),geometry=[],topology=[];
 for(const c of [x,...channels]){const pts=embed(c,lag),d=distances(pts);geometry.push(...Array.from({length:10},(_,i)=>quantile(d,.05+.9*i/9)));topology.push(...betti(pts));}
 const envs=channels.map(envelope),couplings=[];for(let i=0;i<6;i++)for(let j=i+1;j<6;j++)couplings.push(correlation(envs[i],envs[j]));
 return {wave,f0,energy:en,geometry,topology,couplings,combined:[...en.map(v=>v*3),...geometry.map(v=>v*.35),...topology.map(v=>v*.18),...couplings.map(v=>v*.16)]};}
function distFeatures(a,b,key){if(!a||!b)return null;const x=a[key],y=b[key];let sum=0;for(let i=0;i<x.length;i++)sum+=(x[i]-y[i])**2;return Math.sqrt(sum/x.length);}
window.HDRVoice={synth,signature,betti,embed,distFeatures};
window.HDRVoiceApp=app;
})();
