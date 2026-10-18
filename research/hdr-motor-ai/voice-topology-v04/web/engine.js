/* HDR WaveForm v0.4 — 3D membrane+3D acoustic FDTD. MIT.
   True 3D pressure voxels, deformable 3D triangular membrane with vector displacements.
   Reciprocal pressure-force and displacement-flux terms; linearized voxel FSI.
*/
(function(root,factory){const lib=factory();if(typeof module==='object'&&module.exports)module.exports=lib;else root.WaveForm=lib;})(typeof globalThis!=='undefined'?globalThis:this,function(){'use strict';
const SOUND_SPEED=343, TAU=2*Math.PI;
const base=(s)=>({controls:(s?.controls||[.88,1.15,.72,1.08]).slice(),length:s?.length||.168,aspect:s?.aspect||.88,rod:!!s?.rod,lobes:(s?.lobes||[0,0]).slice()});
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
   const t=Math.atan2(zz/rz[x],yy/ry[x]),thickness=1+Math.sin(Math.PI*x/(nx-1))*(shape.lobes[0]*Math.cos(2*t)+shape.lobes[1]*Math.sin(3*t));
   const isFluid=Math.hypot(yy/ry[x],zz/rz[x])<=thickness && (!shape.rod || yy*yy+zz*zz>=.0055**2);
   mask[(x*ny+y)*nz+z]=isFluid?1:0;
  }
 }
 const nphi=16, cell=new Int32Array(nx*nphi),normal=new Float64Array(nx*nphi*3),patch=new Float64Array(nx*nphi),rest=new Float64Array(nx*nphi*3),triangles=[];
 const idx=(x,y,z)=>(x*ny+y)*nz+z;
 for(let x=0;x<nx;x++)for(let j=0;j<nphi;j++){
  const th=j*TAU/nphi,cs=Math.cos(th),sn=Math.sin(th),n=x*nphi+j;
  const envelope=Math.sin(Math.PI*x/(nx-1)),rr=1+envelope*(shape.lobes[0]*Math.cos(2*th)+shape.lobes[1]*Math.sin(3*th));
  const dr=envelope*(-2*shape.lobes[0]*Math.sin(2*th)+3*shape.lobes[1]*Math.cos(3*th));
  const yy=ry[x]*rr*cs,zz=rz[x]*rr*sn,yp=ry[x]*(dr*cs-rr*sn),zp=rz[x]*(dr*sn+rr*cs);
  rest[3*n]=x*dx;rest[3*n+1]=yy;rest[3*n+2]=zz;
  const len=Math.hypot(yp,zp);normal[3*n+1]=zp/len;normal[3*n+2]=-yp/len;
  patch[n]=dx*TAU/nphi*len;
  let found=false;
  for(const factor of [.90,.80,.65,.50,0]){
   const iy=Math.max(0,Math.min(ny-1,Math.round((factor*yy+extent/2)/dy)));
   const iz=Math.max(0,Math.min(nz-1,Math.round((factor*zz+extent/2)/dz)));
   if(mask[idx(x,iy,iz)]){cell[n]=idx(x,iy,iz);found=true;break;}
  }
  if(!found)throw Error('Surface node lacks interior voxel');
 }
 for(let x=0;x<nx-1;x++)for(let j=0;j<nphi;j++){
  const a=x*nphi+j,b=x*nphi+(j+1)%nphi,c=(x+1)*nphi+j,d=(x+1)*nphi+(j+1)%nphi;
  triangles.push(a,c,b,b,c,d);
 }
 return {shape,nx,ny,nz,nphi,N,dx,dy,dz,mask,radial,area,ry,rz,cell,normal,patch,rest,triangles};
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
function simulate(s, {steps=1250,nx=28,ny=15,nz=15,nphi=16,excitation='chirp',frames=14,wall={},drive:givenDrive=null}={}){
 if(nphi!==16)throw Error('Browser engine uses 16 surface sectors');
 const g=generate(s,nx,ny,nz),{N,mask,dx,dy,dz,cell,normal,patch}=g;
 const dt=.40/(SOUND_SPEED*Math.sqrt(1/dx**2+1/dy**2+1/dz**2));
 const cfl=SOUND_SPEED*dt*Math.sqrt(1/dx**2+1/dy**2+1/dz**2),rho=1.2,K=rho*SOUND_SPEED**2,vol=dx*dy*dz;
 const w={enabled:true,mass:.14,stiffness:2.1e6,damping:34,tension:3,loss:8,gain:.35,...wall};
 if(!(w.mass>0&&w.stiffness>0&&w.damping>=0&&w.tension>=0&&w.loss>=0&&w.gain>=0))throw Error('Nonpassive wall parameters');
 const spacing=Math.min(dx,...patch.map(q=>q/dx));
 if(Math.sqrt(w.stiffness/w.mass+8*w.tension/(w.mass*spacing**2))*dt>.23)throw Error('Surface CFL');
 const drive=givenDrive ? Float64Array.from(givenDrive) : driveSignal(steps,dt,excitation);
 if(drive.length!==steps || !drive.every(Number.isFinite))throw Error('Invalid drive');
 const signal=new Float64Array(steps),wallTrace=new Float64Array(steps),asymmetryTrace=new Float64Array(steps),energy=new Float64Array(steps);
 const p=new Float64Array(N),vx=new Float64Array((nx-1)*ny*nz),vy=new Float64Array(nx*(ny-1)*nz),vz=new Float64Array(nx*ny*(nz-1));
 const U=new Float64Array(nx*nphi*3),V=new Float64Array(U.length),flux=new Float64Array(N),pstride=ny*nz;
 const idx=(x,y,z)=>(x*ny+y)*nz+z;
 const snapshots=[],frameN=Math.max(1,frames|0);
 for(let t=0;t<steps;t++){
  for(let x=0;x<nx-1;x++)for(let y=0;y<ny;y++)for(let z=0;z<nz;z++){
   const i=idx(x,y,z),q=i+pstride;vx[i]=mask[i]&&mask[q]?vx[i]-dt/(rho*dx)*(p[q]-p[i]):0;
  }
  for(let x=0;x<nx;x++)for(let y=0;y<ny-1;y++)for(let z=0;z<nz;z++){
   const i=idx(x,y,z),q=i+nz,j=(x*(ny-1)+y)*nz+z;
   vy[j]=mask[i]&&mask[q]?vy[j]-dt/(rho*dy)*(p[q]-p[i]):0;
  }
  for(let x=0;x<nx;x++)for(let y=0;y<ny;y++)for(let z=0;z<nz-1;z++){
   const i=idx(x,y,z),q=i+1,j=(x*ny+y)*(nz-1)+z;
   vz[j]=mask[i]&&mask[q]?vz[j]-dt/(rho*dz)*(p[q]-p[i]):0;
  }
  if(w.enabled){
   flux.fill(0); const prev=U.slice();
   for(let x=1;x<nx-1;x++)for(let j=0;j<nphi;j++){
    const jp=(j+1)%nphi,jm=(j+nphi-1)%nphi,n=x*nphi+j,ni=3*n,ii=cell[n],arcl=Math.max(1e-5,patch[n]/dx),cs=(TAU/nphi/arcl)**2;
    for(let d=0;d<3;d++){
     const ax=(prev[ni-3*nphi+d]-2*prev[ni+d]+prev[ni+3*nphi+d])/(dx*dx);
     const az=(prev[3*(x*nphi+jm)+d]-2*prev[ni+d]+prev[3*(x*nphi+jp)+d])*cs;
     const acc=(normal[ni+d]*p[ii]-w.stiffness*prev[ni+d]-w.damping*V[ni+d]+w.tension*(ax+az))/w.mass;
     V[ni+d]+=dt*acc;
    }
   }
   for(let x=1;x<nx-1;x++)for(let j=0;j<nphi;j++){
    const n=x*nphi+j,ni=3*n;let dv=0;
    for(let d=0;d<3;d++){U[ni+d]+=dt*V[ni+d];dv+=V[ni+d]*normal[ni+d];}
    flux[cell[n]]+=patch[n]/vol*dv;
   }
  }
  for(let x=0;x<nx;x++)for(let y=0;y<ny;y++)for(let z=0;z<nz;z++){
   const i=idx(x,y,z);if(!mask[i])continue;let div=0;
   if(x<nx-1)div+=vx[i]/dx;if(x>0)div-=vx[i-pstride]/dx;
   if(y<ny-1)div+=vy[(x*(ny-1)+y)*nz+z]/dy;
   if(y>0)div-=vy[(x*(ny-1)+y-1)*nz+z]/dy;
   if(z<nz-1)div+=vz[(x*ny+y)*(nz-1)+z]/dz;
   if(z>0)div-=vz[(x*ny+y)*(nz-1)+z-1]/dz;
   p[i]=(1-w.loss*dt)*p[i]-K*dt*(div+(w.enabled?flux[i]:0))+(x===2?w.gain*drive[t]:0);
  }
  let sum=0,num=0;for(let y=0;y<ny;y++)for(let z=0;z<nz;z++)if(mask[idx(nx-4,y,z)]){sum+=p[idx(nx-4,y,z)];num++;}
  signal[t]=sum/Math.max(1,num);
  const mid=Math.floor(nx/2);let radial=0,asym=0;
  for(let j=0;j<nphi;j++)for(let d=0;d<3;d++)radial+=U[3*(mid*nphi+j)+d]*normal[3*(mid*nphi+j)+d];
  asym=U[3*(mid*nphi)+1]-U[3*(mid*nphi+nphi/2)+1];
  wallTrace[t]=radial/nphi;asymmetryTrace[t]=asym;
  let E=0;for(let i=0;i<N;i++)if(mask[i])E+=.5*p[i]*p[i]/K*vol;
  for(let i=0;i<vx.length;i++)E+=.5*rho*vx[i]*vx[i]*vol;
  for(let i=0;i<vy.length;i++)E+=.5*rho*vy[i]*vy[i]*vol;
  for(let i=0;i<vz.length;i++)E+=.5*rho*vz[i]*vz[i]*vol;
  if(w.enabled)for(let x=1;x<nx-1;x++)for(let j=0;j<nphi;j++){
   const n=x*nphi+j,ni=3*n,im=3*(x*nphi+(j+nphi-1)%nphi),ix=ni-3*nphi,dl=Math.max(1e-5,patch[n]/dx);
   for(let d=0;d<3;d++){
    E+=.5*patch[n]*(w.mass*V[ni+d]**2+w.stiffness*U[ni+d]**2);
    E+=.5*w.tension*patch[n]*((U[ni+d]-U[ix+d])**2/dx**2+(U[ni+d]-U[im+d])**2/dl**2);
   }
  }
  energy[t]=E;
  if(t===steps-1 || snapshots.length<frameN && t>=Math.floor((snapshots.length+1)*steps/(frameN+1))-1){
   const panel=new Float32Array(nx*ny),disp=new Float32Array(U.length),z=Math.floor(nz/2);
   disp.set(U);for(let x=0;x<nx;x++)for(let y=0;y<ny;y++)panel[x*ny+y]=p[idx(x,y,z)];
   snapshots.push({step:t,panel,wall:disp});
  }
 }
 let maxWall=0;for(let n=0;n<U.length/3;n++)maxWall=Math.max(maxWall,Math.hypot(U[3*n],U[3*n+1],U[3*n+2]));
 if(!Number.isFinite(maxWall)||maxWall>Math.min(dx,dy,dz)*.07)throw Error('Linear voxel FSI limit exceeded');
 return {signal,drive,wallTrace,asymmetryTrace,energy,maxWall,dt,geometry:g,frames:snapshots,cfl,excitation,wall:w,displacement:U,velocity:V};
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
 const shape=base(s),wall=options.wall||{},stepOrder=[.25,.14,.075,.04].slice(0,iterations);
 let cur=simulate(shape,{steps:options.steps||1300,frames:1,wall});let best=loss(signature(cur),target),before=best,evals=1;
 for(const step of stepOrder){
  for(let i=0;i<6;i++){
   const lobe=i>=4,index=lobe?i-4:i;
   let win=lobe?shape.lobes[index]:shape.controls[index],wl=best;
   for(const direction of [-1,1]){
    const controls=shape.controls.slice(),lobes=shape.lobes.slice();
    if(lobe)lobes[index]=Math.max(-.25,Math.min(.25,lobes[index]+step*direction));
    else controls[index]=Math.max(.5,Math.min(1.48,controls[index]+step*direction));
    const v=loss(signature(simulate({...shape,controls,lobes},{steps:options.steps||1300,frames:1,wall})),target);evals++;
    if(v<wl){win=lobe?lobes[index]:controls[index];wl=v;}
   }
   if(lobe)shape.lobes[index]=win;else shape.controls[index]=win;
   best=wl;onProgress({...shape,controls:shape.controls.slice(),lobes:shape.lobes.slice(),before,after:best,evals});
   await new Promise(resolve=>setTimeout(resolve,0));
  }
 }
 return {shape,before,after:best,evals};
}
return {base,generate,simulate,signature,loss,learn};
});
