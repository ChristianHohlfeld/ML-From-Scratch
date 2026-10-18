"""Cross-implementation verification: Python Numba and browser JS solve same PDE."""
import json
import subprocess
from pathlib import Path
import numpy as np
from waveform import Shape,propagate

ROOT=Path(__file__).resolve().parents[1]

def test_js_python_same_wave_equation():
    shape=Shape((.81,1.12,.88,1.25))
    n=500
    py=propagate(shape,steps=n,nx=24,ny=15,nz=15)
    code=f'''const W=require({str(ROOT/'web/engine.js')!r});
    const r=W.simulate({{controls:[.81,1.12,.88,1.25],length:.168,aspect:.88,rod:false}},
    {{steps:{n},nx:24,ny:15,nz:15,frames:1}});
    process.stdout.write(JSON.stringify({{dt:r.dt,y:Array.from(r.signal)}}));'''
    out=subprocess.run(['node','-e',code],capture_output=True,text=True,check=True)
    js=json.loads(out.stdout)
    assert abs(js['dt']-py['dt'])<1e-14
    np.testing.assert_allclose(js['y'],py['signal'],rtol=5e-7,atol=1e-7)
