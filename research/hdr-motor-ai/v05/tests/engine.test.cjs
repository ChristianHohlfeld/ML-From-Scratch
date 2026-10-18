const assert=require('node:assert/strict');
const {test}=require('node:test');
const G=require('../web/engine.js');
const D={base:[.3,-.1],gears:[{k:1,a:[.8,.2]},{k:-2,a:[.12,-.18]},{k:5,a:[.2,.09]}]};
test('single shaft gears preserve integer ratios',()=>{
 const m=new G.Motor(D);for(let i=0;i<50;i++)m.advance(.01,1.7);
 assert.equal(m.read().length,2);
 assert(Math.abs(m.angle)>0.01);
});
test('closed gear trajectory',()=>{
 const a=G.tip(D,0),b=G.tip(D,2*Math.PI);
 assert(Math.hypot(a[0]-b[0],a[1]-b[1])<1e-10);
});
test('offline fitting recovers known topology',()=>{
 const phases=Array.from({length:120},(_,i)=>G.TAU*i/120);
 const data=phases.map(p=>G.tip(D,p)), f=G.fit(phases,data,[1,-2,5]);
 assert(G.mse(f,phases,data)<1e-12);
});
test('topology search selects signed gears',()=>{
 const target=G.trajectory(D,128),s=new G.Search(target,7,1e-7);s.learn();
 assert.deepEqual(s.ks.slice().sort((a,b)=>a-b),[-2,1,5]);
 assert(G.mse(s.design,Array.from({length:128},(_,i)=>G.TAU*i/128),target)<1e-10);
});
test('install does not reset physical shaft phase',()=>{
 const m=new G.Motor(D);for(let i=0;i<40;i++)m.advance(.01,1);
 const angle=m.angle; m.install({base:[0,0],gears:D.gears.slice(0,2)});
 assert.equal(m.angle,angle);
});
test('torque input is capped',()=>{
 const m=new G.Motor(D);m.limit=.001;m.advance(.01,10000);
 assert(Math.abs(m.speed)<=.01*.001/m.inertia+1e-12);
});
