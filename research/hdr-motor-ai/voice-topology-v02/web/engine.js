/* HDR WaveForm v0.2 — 3D scalar acoustic FDTD. MIT.
   True 3D pressure voxels, deformable elliptic volume (quasi-static walls).
   Zero-flux wall neighbor coupling, 3D CFL bound, no signal-to-3D projection.
*/
(function(root,factory){const lib=factory();if(typeof module==='object'&&module.exports)module.exports=lib;else root.WaveForm=lib;})(typeof globalThis!=='undefined'?globalThis:this,function(){'use strict';
const SOUND_SPEED=343, TAU=2*Math.PI;
const base=(s)=>({controls:(s?.controls||[.88,1.15,.72,1.08]).slice(),length:s?.length||.168,aspect:s?.aspect||.88,rod:!!s?.rod});
function generate(s={},nx=34,ny=19,nz=19,extent=.084){
 const shape=base(s); const dx=shape.length/(nx-1),dy=extent/(ny-1),dz=extent/(nz-1),N=nx*ny*nz;
 const mask=new Uint8Array(N), radial=new Float64Array(nx), area=new Float64Array(nx), ry=new Float64Array(nx), rz=new Float64Array(nx);
 const points=[1,...shape.controls,1];
 for(let x=0;x<nx;x++){
  let u=5*x/(nx-1),j=Math.min(4,Math.floor(u));radial[x]=points[j]*(1-(u-j))+points[j+1]*(u-j);
 }
 const smooth=radial.slice(); for(let x=0;x<nx;x++)radial[x]=(smooth[Math.max(0,x-2)]+4*smooth[Math.max(0,x-1)]+6*smooth[x]+4*smooth[Math.min(nx-1,x+1)]+smooth[Math.min(nx-1,x+2)])/16;
 for(let x=0;x<nx;x++){
  ry[x]=.019*radial[x];rz[x]=ry[x]*shape.aspect;area[x]=Math.PI*ry[x]*rz[x];
  for(let y=0;y<ny;y++)for(let z=0;z<nz;z++){
   const yy=-extent/2+y*dy,zz=-extent/2+z*dz;
   const isFluid=(yy/ry[x])**2+(zz/rz[x])**2<=1 && (!shape.rod || yy*yy+zz*zz>=.0055**2);
   mask[(x*ny+y)*nz+z]=isFluid?1:0;
  }
 }
 return {shape,nx,ny,nz,N,dx,dy,dz,mask,radial,area,ry,rz};
}
function driveSignal(steps,dt,variant='chirp'){
 const x=new Float64Array(steps),last=(steps-1)*dt;
 for(let n=0;n<steps;n++){
  const t=n*dt,fade=Math.sin(Math.PI*n/(steps-1))**2;
  if(variant==='chirp')x[n]=fade*Math.sin(TAU*(330*t+.5*(3900-330)/last*t*t));
  else if(variant==='multitone')x[n]=fade*(Math.sin(TAU*470*t)+.6*Math.sin(TAU*1160*t+.4)+.45*Math.sin(TAU*2520*t+.8));
  else if(variant==='impulse')x[n]=Math.exp(-.5*((n-Math.floor(steps/14))/10)**2);
  else throw Error('unknown excitation');
 }
 return x;
}
function simulate(s, {steps=1700, nx=34,ny=19,nz=19,excitation='chirp',frames=16}={}){
 const g=generate(s,nx,ny,nz),{N,mask}=g;
 const dt=.78/(SOUND_SPEED*Math.sqrt(1/g.dx**2+1/g.dy**2+1/g.dz**2));
 const lx=(SOUND_SPEED*dt/g.dx)**2,ly=(SOUND_SPEED*dt/g.dy)**2,lz=(SOUND_SPEED*dt/g.dz)**2;
 if(lx+ly+lz>=1)throw Error('CFL unstable');
 const drive=driveSignal(steps,dt,excitation),signal=new Float64Array(steps),snapshots=[];
 let prev=new Float64Array(N),p=new Float64Array(N),nxt=new Float64Array(N);
 const strideY=nz,strideX=ny*nz,src=2,probe=nx-4,planeX=Math.floor(nx/2),centerZ=Math.floor(nz/2);
 const frameEvery=Math.max(1,Math.floor(steps/frames));
 for(let t=0;t<steps;t++){
  for(let x=1;x<nx-1;x++)for(let y=1;y<ny-1;y++)for(let z=1;z<nz-1;z++){
   const i=(x*ny+y)*nz+z;
   if(!mask[i]){nxt[i]=0;continue;}
   const v=p[i];let lap=0;
   if(mask[i+strideX])lap+=lx*(p[i+strideX]-v);
   if(mask[i-strideX])lap+=lx*(p[i-strideX]-v);
   if(mask[i+strideY])lap+=ly*(p[i+strideY]-v);
   if(mask[i-strideY])lap+=ly*(p[i-strideY]-v);
   if(mask[i+1])lap+=lz*(p[i+1]-v);
   if(mask[i-1])lap+=lz*(p[i-1]-v);
   const d=.003*(x<4||x>nx-5?9:1);
   nxt[i]=(2-d)*v-(1-d)*prev[i]+lap+(x===src?drive[t]*.035:0);
  }
  let total=0,valid=0;
  for(let y=1;y<ny-1;y++)for(let z=1;z<nz-1;z++){
   const i=(probe*ny+y)*nz+z;if(mask[i]){total+=nxt[i];valid++;}
  }
  signal[t]=total/Math.max(1,valid);
  if(t%frameEvery===0||t===steps-1){
   const panel=new Float32Array(nx*ny);
   for(let x=0;x<nx;x++)for(let y=0;y<ny;y++) panel[x*ny+y]=nxt[(x*ny+y)*nz+centerZ];
   snapshots.push({step:t,panel});
  }
  [prev,p,nxt]=[p,nxt,prev];
 }
 return {signal,drive,dt,geometry:g,frames:snapshots,cfl:Math.sqrt(lx+ly+lz),excitation};
}
function dft(signal,dt,f){let re=0,im=0;const n=signal.length;const step=TAU*f*dt;for(let i=0;i<n;i++){
 const win=.5-.5*Math.cos(TAU*i/(n-1)),a=step*i;re+=signal[i]*win*Math.cos(a);im-=signal[i]*win*Math.sin(a);
}return Math.hypot(re,im);}
function signature(r,bins=28){
 const fv=new Float64Array(bins),y=new Float64Array(bins),x=new Float64Array(bins);
 let xmax=0;
 for(let k=0;k<bins;k++){
  fv[k]=200*Math.pow(3900/200,k/(bins-1));x[k]=dft(r.drive,r.dt,fv[k]);y[k]=dft(r.signal,r.dt,fv[k]);xmax=Math.max(xmax,x[k]);
 }
 let norm=0;for(let k=0;k<bins;k++){
  y[k]=Math.log1p(50*y[k]*x[k]/(x[k]**2+(.045*xmax)**2));norm+=y[k]*y[k];
 }
 norm=Math.sqrt(norm)||1;for(let k=0;k<bins;k++)y[k]/=norm;
 return Array.from(y);
}
const loss=(a,b)=>a.reduce((s,v,i)=>s+(v-b[i])**2,0)/a.length;
async function learn(target,s,options={}){
 const onProgress=options.onProgress||(()=>{}),iterations=options.iterations||3;
 const shape=base(s),stepOrder=[.25,.14,.075,.04].slice(0,iterations);
 let cur=simulate(shape,{steps:options.steps||1300,frames:1});let best=loss(signature(cur),target),before=best,evals=1;
 for(const step of stepOrder){
  for(let i=0;i<4;i++){
   let win=shape.controls[i],wl=best;
   for(const direction of [-1,1]){
    const controls=shape.controls.slice();controls[i]=Math.max(.5,Math.min(1.48,controls[i]+step*direction));
    const v=loss(signature(simulate({...shape,controls},{steps:options.steps||1300,frames:1})),target);evals++;
    if(v<wl){win=controls[i];wl=v;}
   }
   shape.controls[i]=win;best=wl;onProgress({...shape,controls:shape.controls.slice(),before,after:best,evals});
   await new Promise(resolve=>setTimeout(resolve,0));
  }
 }
 return {shape,before,after:best,evals};
}
return {base,generate,simulate,signature,loss,learn};
});
