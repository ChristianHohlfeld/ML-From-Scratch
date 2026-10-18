/* HDR Gear AI v0.6 — deterministic two-input compound gear engine, MIT.
   Structural learning is OFFLINE numeric Fourier projection. Runtime only
   advances physical motor state and reads k*A+l*B linkage coordinates. */
(function(root,factory){const api=factory();if(typeof module!=='undefined'&&module.exports)module.exports=api;root.HDRGear=api})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const PI=Math.PI,TAU=2*PI;
const truth={
 compound:{name:'Gekoppelt',origin:[.05,.08],stages:[{k:1,l:0,a:[.23,.16]},{k:0,l:-2,a:[.14,-.2]},{k:2,l:-1,a:[.42,.09]},{k:-1,l:2,a:[.2,.15]},{k:1,l:1,a:[.16,-.08]}]},
 additive:{name:'Additiv',origin:[-.07,.04],stages:[{k:1,l:0,a:[.23,.16]},{k:-2,l:0,a:[.1,-.2]},{k:0,l:1,a:[.38,.1]},{k:0,l:-2,a:[.12,.09]}]},
 sharp:{name:'Nicht harmonisch',origin:[0,0],stages:[]}
};
function sample(model,A,B){let x=model.origin[0],y=model.origin[1];for(const s of model.stages){const p=s.k*A+s.l*B,c=Math.cos(p),z=Math.sin(p);x+=s.a[0]*c-s.a[1]*z;y+=s.a[0]*z+s.a[1]*c;}return [x,y]}
function target(task,A,B){if(task==='sharp'){const v=Math.tanh(4*Math.sin(A)*Math.sin(B)),u=Math.tanh(5*Math.cos(A-B));return [.3*v+.08*Math.sin(2*A),.28*u+.09*Math.cos(B)]}return sample(truth[task],A,B)}
function testErr(model,task,n=13){let e=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++){const A=-PI+TAU*(i+.37)/n,B=-PI+TAU*(j+.23)/n,p=sample(model,A,B),q=target(task,A,B);e+=(p[0]-q[0])**2+(p[1]-q[1])**2;}return e/(n*n)}
function featureCoeff(task,k,l,n=20){let x=0,y=0;for(let i=0;i<n;i++)for(let j=0;j<n;j++){const A=-PI+TAU*i/n,B=-PI+TAU*j/n,[u,v]=target(task,A,B);const p=k*A+l*B,c=Math.cos(p),s=Math.sin(p);x+=u*c+v*s;y+=v*c-u*s}return [x/(n*n),y/(n*n)]}
function fit(task,ratios){const c=featureCoeff(task,0,0),s=ratios.map(([k,l])=>({k,l,a:featureCoeff(task,k,l)}));return {format:'hdr-gear-ai/2',origin:c,stages:s}}
function learnOne(machine,task,{maxRatio=3,allowMixed=true,maxStages=10,penalty=.0001}={}){
 if(machine.stages.length>=maxStages)return {machine,changed:false,event:'Budget erreicht'};
 const used=new Set(machine.stages.map(s=>s.k+','+s.l));let best=null;
 for(let k=-maxRatio;k<=maxRatio;k++)for(let l=-maxRatio;l<=maxRatio;l++){
  if(!(k||l)||(!allowMixed&&k&&l)||used.has(k+','+l))continue;
  const a=featureCoeff(task,k,l),amp=a[0]**2+a[1]**2;
  if(!best||amp>best.amp)best={k,l,amp};
 }
 if(!best||best.amp<=penalty)return {machine,changed:false,event:'Keine relevante Stufe'};
 const ks=[...machine.stages.map(s=>[s.k,s.l]),[best.k,best.l]];
 const next=fit(task,ks),prev=testErr(machine,task),e=testErr(next,task);
 if(e>=prev-penalty)return {machine,changed:false,event:'Kein Gewinn'};
 return {machine:next,changed:true,event:`Getriebe (${best.k}, ${best.l}) ergänzt`,gain:prev-e};
}
function prune(machine,task,{penalty=.0001}={}){let best=null,base=testErr(machine,task);for(let i=0;i<machine.stages.length;i++){const ratios=machine.stages.filter((_,j)=>j!==i).map(s=>[s.k,s.l]),m=fit(task,ratios);const e=testErr(m,task);if(e-base<penalty&&(!best||e<best.err))best={machine:m,err:e,index:i}};return best?{machine:best.machine,changed:true,event:'Schwache Stufe entfernt'}:{machine,changed:false,event:'Alle Stufen tragen bei'}}
class Drive{
 constructor(machine){this.machine=machine;this.a=[0,0];this.v=[0,0];this.targets=[0,0];this.kp=24;this.kd=9;this.drag=.10;this.maxTorque=40;this.mass=null;this.massMachine=null}
 inertiaMatrix(){let aa=1,bb=1,ab=0;for(const s of this.machine.stages){const J=.02*(s.a[0]**2+s.a[1]**2);aa+=J*s.k*s.k;bb+=J*s.l*s.l;ab+=J*s.k*s.l}return [aa,ab,bb]}
 step(dt){dt=Math.max(0,Math.min(.025,dt));const tau=[];for(let j=0;j<2;j++)tau[j]=Math.max(-this.maxTorque,Math.min(this.maxTorque,this.kp*(this.targets[j]-this.a[j])-this.kd*this.v[j]))-this.drag*this.v[j];
 const [aa,ab,bb]=this.inertiaMatrix(),det=aa*bb-ab*ab,acc=[(bb*tau[0]-ab*tau[1])/det,(aa*tau[1]-ab*tau[0])/det];
 for(let j=0;j<2;j++){this.v[j]+=dt*acc[j];this.a[j]+=dt*this.v[j]}return this.read()}
 read(){return sample(this.machine,...this.a)}
 gearAngle(s){return s.k*this.a[0]+s.l*this.a[1]+Math.atan2(s.a[1],s.a[0])}
}
function decode(obj){if(obj.format!=='hdr-gear-ai/2'||!Array.isArray(obj.origin)||obj.origin.length!==2||!Array.isArray(obj.stages))throw Error('Invalid model');const ks=new Set();for(const s of obj.stages){if(!Number.isInteger(s.k)||!Number.isInteger(s.l)||!(s.k||s.l)||!Array.isArray(s.a)||s.a.length!==2||s.a.some(v=>!Number.isFinite(v)))throw Error('Invalid stage');const key=s.k+','+s.l;if(ks.has(key))throw Error('Repeated ratio');ks.add(key)}if(obj.origin.some(v=>!Number.isFinite(v)))throw Error('Invalid origin');return JSON.parse(JSON.stringify(obj))}
return {PI,TAU,truth,sample,target,testErr,featureCoeff,fit,learnOne,prune,Drive,decode};
});
