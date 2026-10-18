const W=require('../web/engine.js');
const s={controls:[.88,1.15,.72,1.08],length:.168,aspect:.88,rod:false};
const r=W.simulate(s,{steps:360,nx:24,ny:13,nz:13,frames:2});
console.log(JSON.stringify({signal:Array.from(r.signal),wall:Array.from(r.displacement),energy:Array.from(r.energy),max:r.maxWall,dt:r.dt,vertices:Array.from(r.geometry.rest),normals:Array.from(r.geometry.normal)}));
