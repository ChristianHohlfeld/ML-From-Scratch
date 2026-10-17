/* HDR Motor AI v0.3 - exact mechanical network in ordinary JavaScript.
 * No CDN, no frameworks, no optimizer, no periodic recalculation of V0.
 * Can run in browser or Node.js. Parameter units: radians, seconds, normalized torque.
 */
(function (root) {
'use strict';
const KIND_K = {pair:0.3, neighbor:0.3, group:0.3, hub:0.3};
const zero = n => Array(n).fill(0);
function createDefault(n=6, groups=2, channels=3) {
  if (n < 2 || groups < 1 || groups > n || channels < 1 || n > 18 || channels > 12) throw Error('Invalid topology');
  const edges=[];
  function add(a,b,kind,orientation=1) {edges.push({a,b,kind,k:Array(channels).fill(KIND_K[kind]),cubic:Array(channels).fill(0.04),ratio:Array(channels).fill(orientation)});}
  for(let a=0;a<n;a++)for(let b=a+1;b<n;b++)add(a,b,'pair');
  for(let a=0;a<n-1;a++)add(a,a+1,'neighbor');
  for(let a=0;a<n;a++)add(a,n+a%groups,'group');
  for(let g=0;g<groups;g++)add(n+g,n+groups,'hub',g%2?-1:1);
  return {schema:'hdr-motor-v0.3',n,groups,channels,dt:.035,inertia:1,damping:2.4,motorStiffness:2.2,edges,
    gain:Array.from({length:n},()=>Array(channels).fill(1)),bias:Array.from({length:n},()=>zero(channels)),
    readoutWeight:Array.from({length:channels},(_,i)=> i===0?1:0),readoutBias:0};
}
function validated(model) {
  if (!model || model.schema!=='hdr-motor-v0.3' || !Number.isInteger(model.n)|| !Number.isInteger(model.groups)||!Number.isInteger(model.channels))throw Error('Unsupported model schema');
  const {n,groups,channels}=model;
  if(n<2||n>18||groups<1||groups>n||channels<1||channels>12) throw Error('Model too large or invalid');
  for(const key of ['dt','damping','inertia','motorStiffness'])if(!Number.isFinite(model[key])||model[key]<=0||model[key]>20)throw Error('Invalid '+key);
  if(model.dt>0.05)throw Error('Unsafe integration step');
  const total=n+groups+1;
  function vec(v, len, lo,hi,lab){if(!Array.isArray(v)||v.length!==len ||v.some(x=>!Number.isFinite(x)||x<lo||x>hi))throw Error('Invalid '+lab);}
  if(!Array.isArray(model.edges)||model.edges.length>300)throw Error('Too many edges');
  for(const [i,e] of model.edges.entries()){
    if(!Number.isInteger(e.a)||!Number.isInteger(e.b)||e.a===e.b||e.a<0||e.a>=total||e.b<0||e.b>=total||!['pair','neighbor','group','hub'].includes(e.kind))throw Error('Edge '+i+' invalid');
    vec(e.k,channels,1e-5,20,'stiffness');vec(e.cubic,channels,0,2,'cubic');vec(e.ratio,channels,-2,2,'ratio');
    if(e.ratio.some(x=>Math.abs(x)<0.1))throw Error('Zero gearing');
  }
  if(model.gain?.length!==n||model.bias?.length!==n)throw Error('Drive calibration shape mismatch');
  model.gain.forEach(v=>vec(v,channels,-5,5,'gain'));
  model.bias.forEach(v=>vec(v,channels,-5,5,'bias'));
  vec(model.readoutWeight,channels,-10,10,'readout');
  if(!Number.isFinite(model.readoutBias)||Math.abs(model.readoutBias)>10)throw Error('Invalid readout bias');
  return model;
}
function makeState(m) { validated(m);const total=m.n+m.groups+1;return {angle:Array.from({length:total},()=>zero(m.channels)),velocity:Array.from({length:total},()=>zero(m.channels)),time:0};}
function checkState(m,s,targets){if(!s ||s.angle?.length!==m.n+m.groups+1||s.velocity?.length!==s.angle.length||targets.length!==m.n||targets.some(v=>!Number.isFinite(v)||Math.abs(v)>5))throw Error('Invalid state or input');}
function step(m,s,targets,dt=m.dt){checkState(m,s,targets);if(!Number.isFinite(dt)||dt<=0||dt>.05)throw Error('Unsafe dt');
  const total=m.n+m.groups+1, C=m.channels;
  const torque=Array.from({length:total},()=>zero(C));
  for(const e of m.edges)for(let c=0;c<C;c++){
    const ext=s.angle[e.a][c]+e.ratio[c]*s.angle[e.b][c];
    const stress=e.k[c]*ext+e.cubic[c]*ext*ext*ext;
    torque[e.a][c]-=stress;torque[e.b][c]-=e.ratio[c]*stress;
  }
  for(let i=0;i<m.n;i++)for(let c=0;c<C;c++)torque[i][c]+=m.motorStiffness*(targets[i]*m.gain[i][c]+m.bias[i][c]-s.angle[i][c]);
  for(let i=0;i<total;i++)for(let c=0;c<C;c++){
    const acceleration=(torque[i][c]-m.damping*s.velocity[i][c])/m.inertia;
    s.velocity[i][c]+=dt*acceleration;
    s.angle[i][c]+=dt*s.velocity[i][c];
  }
  s.time+=dt;return s;
}
function evolve(m,s,targets,steps=1){if(!Number.isInteger(steps)||steps<0||steps>100000)throw Error('Invalid steps');for(let i=0;i<steps;i++)step(m,s,targets);return s;}
function read(m,s){const v=s.angle[m.n+m.groups];return m.readoutBias+v.reduce((acc,angle,c)=>acc+angle*m.readoutWeight[c],0);}
function energy(m,s,targets){checkState(m,s,targets);let E=0;
 for(let i=0;i<s.angle.length;i++)for(let c=0;c<m.channels;c++)E+=.5*m.inertia*s.velocity[i][c]**2;
 for(const e of m.edges)for(let c=0;c<m.channels;c++){const x=s.angle[e.a][c]+e.ratio[c]*s.angle[e.b][c];E+=.5*e.k[c]*x*x+.25*e.cubic[c]*x**4;}
 for(let i=0;i<m.n;i++)for(let c=0;c<m.channels;c++){const d=targets[i]*m.gain[i][c]+m.bias[i][c]-s.angle[i][c];E+=.5*m.motorStiffness*d*d;}
 return E;
}
function resize(model,state,targets,newN){
 if(newN<2||newN>18)throw Error('Unsupported motor count');
 const next=createDefault(newN,Math.min(model.groups,newN),model.channels);
 for(const k of ['dt','damping','inertia','motorStiffness'])next[k]=model[k];
 next.readoutWeight=[...model.readoutWeight];next.readoutBias=model.readoutBias;
 const oldN=model.n, G=model.groups;
 const mapIndex=(i)=>i<oldN?i:(i<oldN+G?newN+i-oldN:newN+G);
 const byKey=new Map(next.edges.map(e=>[[e.kind,e.a,e.b].join(':'),e]));
 for(const e of model.edges){if((e.a<oldN && e.a>=newN)||(e.b<oldN&&e.b>=newN))continue;
  const a=mapIndex(e.a),b=mapIndex(e.b),k=[e.kind,a,b].join(':');const f=byKey.get(k);
  if(f){f.k=[...e.k];f.cubic=[...e.cubic];f.ratio=[...e.ratio];}
 }
 for(let i=0;i<Math.min(oldN,newN);i++){next.gain[i]=[...model.gain[i]];next.bias[i]=[...model.bias[i]];}
 const ns=makeState(next);
 for(let i=0;i<oldN+G+1;i++){
  const index=mapIndex(i);
  if(index<ns.angle.length && (i>=oldN || i<newN)){
   ns.angle[index]=[...state.angle[i]];ns.velocity[index]=[...state.velocity[i]];
  }
 }
 ns.time=state.time;
 const inputs=targets.slice(0,newN);while(inputs.length<newN)inputs.push(0);
 return {model:next,state:ns,targets:inputs};
}
const api={createDefault,validated,makeState,step,evolve,read,energy,resize};
root.HDRMotorEngine=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
