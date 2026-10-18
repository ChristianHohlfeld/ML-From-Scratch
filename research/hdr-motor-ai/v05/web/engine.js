/* HDR Gear AI v0.5. One input shaft, signed integer ratio gears, endpoint = V0.
   Structural search is OFFLINE; runtime tick does not solve a fit problem. */
(function(root, factory){
  if(typeof module==='object' && module.exports)module.exports=factory();
  else root.HDRGear=factory();
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const TAU=2*Math.PI;
  function tip(design,p){let x=design.base[0],y=design.base[1];
    for(const g of design.gears){const a=g.k*p;x+=g.a[0]*Math.cos(a)-g.a[1]*Math.sin(a);y+=g.a[0]*Math.sin(a)+g.a[1]*Math.cos(a);}
    return [x,y];}
  function trajectory(design,n){return Array.from({length:n},(_,i)=>tip(design,TAU*i/n));}
  function solve(a,b){const n=b.length,M=a.map((r,i)=>[...r,b[i]]);
    for(let col=0;col<n;col++){let pivot=col;for(let i=col+1;i<n;i++)if(Math.abs(M[i][col])>Math.abs(M[pivot][col]))pivot=i;
      [M[col],M[pivot]]=[M[pivot],M[col]];
      if(Math.abs(M[col][col])<1e-12)throw Error('Singular structural fit');
      const v=M[col][col];for(let k=col;k<=n;k++)M[col][k]/=v;
      for(let i=0;i<n;i++)if(i!==col){const v=M[i][col];for(let k=col;k<=n;k++)M[i][k]-=v*M[col][k];}}
    return M.map(r=>r[n]);}
  function fit(phases,pts,ks){const cols=2+2*ks.length, A=Array.from({length:cols},()=>new Float64Array(cols)),B=new Float64Array(cols);
    for(let i=0;i<phases.length;i++){
      const p=phases[i], r1=new Float64Array(cols),r2=new Float64Array(cols);r1[0]=1;r2[1]=1;
      ks.forEach((k,j)=>{const c=Math.cos(k*p),s=Math.sin(k*p);r1[2+2*j]=c;r1[3+2*j]=-s;r2[2+2*j]=s;r2[3+2*j]=c;});
      for(const [r,y] of [[r1,pts[i][0]],[r2,pts[i][1]]]){
        for(let a=0;a<cols;a++){B[a]+=r[a]*y;for(let b=a;b<cols;b++)A[a][b]+=r[a]*r[b];}}
    }
    for(let a=0;a<cols;a++){for(let b=0;b<a;b++)A[a][b]=A[b][a];A[a][a]+=a<2?1e-9:1e-7;}
    const v=solve(A,Array.from(B));return {base:[v[0],v[1]],gears:ks.map((k,j)=>({k,a:[v[2+2*j],v[3+2*j]]}))};}
  function mse(d,phases,pts){let sum=0;for(let i=0;i<phases.length;i++){
    const p=tip(d,phases[i]);sum+=(p[0]-pts[i][0])**2+(p[1]-pts[i][1])**2;
  }return sum/(2*phases.length);}
  class Search{
    constructor(target,maxRatio=12,penalty=.00004){this.maxRatio=maxRatio;this.penalty=penalty;this.target=target;
      // Interleaved train/validation samples on a fixed continuous target curve.
      this.train=[],this.val=[],this.trainP=[],this.valP=[];
      target.forEach((p,i)=>{const x=TAU*i/target.length;
        if(i%4===0){this.val.push(p);this.valP.push(x);}else{this.train.push(p);this.trainP.push(x);}});
      this.ks=[];this.design=fit(this.trainP,this.train,this.ks);this.history=[];this.done=false;}
    eval(ks){let d=fit(this.trainP,this.train,ks);let e=mse(d,this.valP,this.val);return {d,e,score:e+ks.length*this.penalty};}
    add(){const before=this.eval(this.ks),options=[];
      for(let k=-this.maxRatio;k<=this.maxRatio;k++){if(!k||this.ks.includes(k))continue;
        const z=this.eval(this.ks.concat(k));options.push({k,...z});}
      options.sort((a,b)=>a.score-b.score||Math.abs(a.k)-Math.abs(b.k)||a.k-b.k);
      if(!options.length||options[0].score>=before.score-1e-10){this.done=true;return null;}
      const winner=options[0];this.ks.push(winner.k);this.design=winner.d;
      this.history.push({op:'add',k:winner.k,error:winner.e});return winner.k;}
    prune(){let change=false;for(const k of [...this.ks]){
      const c=this.eval(this.ks.filter(g=>g!==k)),old=this.eval(this.ks);
      if(c.score<old.score-1e-10){this.ks=this.ks.filter(g=>g!==k);this.design=c.d;
        this.history.push({op:'prune',k,error:c.e});change=true;break;}}
      return change;}
    learn(){for(let i=0;i<this.maxRatio*2;i++)if(this.add()===null)break;
      while(this.prune()){};return this.design;}
  }
  class Motor{
    constructor(design){this.design=design;this.angle=0;this.speed=0;this.kp=14;this.damping=7;this.limit=35;this.friction=.06;this.inertia=1+design.gears.reduce((sum,g)=>sum+.012*g.k*g.k*(g.a[0]*g.a[0]+g.a[1]*g.a[1]),0);}
    advance(dt,targetSpeed){if(!(dt>0&&dt<=.05))throw Error('dt');const t=Math.max(-this.limit,Math.min(this.limit,this.kp*(targetSpeed-this.speed)));
      this.speed+=dt*(t-this.friction*this.speed)/this.inertia;this.angle+=dt*this.speed;return this.read();}
    read(){return tip(this.design,this.angle);}
    install(d){this.design=d;this.inertia=1+d.gears.reduce((s,g)=>s+.012*g.k*g.k*(g.a[0]*g.a[0]+g.a[1]*g.a[1]),0);}
  }
  function target(name,n){const p=Array.from({length:n},(_,i)=>TAU*i/n);
    if(name==='mechanical')return trajectory({base:[.08,-.1],gears:[{k:1,a:[.70,.1]},{k:-2,a:[.23,.12]},{k:4,a:[-.20,.04]},{k:7,a:[.08,-.09]}]},n);
    if(name==='star')return p.map(t=>{const r=.7+.23*Math.cos(5*t);return [r*Math.cos(t),r*Math.sin(t)];});
    return p.map(t=>[Math.sign(Math.sin(t))*Math.abs(Math.sin(t))**.33,Math.sign(Math.cos(t))*Math.abs(Math.cos(t))**.33]);
  }
  return {TAU,tip,trajectory,fit,mse,Search,Motor,target};
});
