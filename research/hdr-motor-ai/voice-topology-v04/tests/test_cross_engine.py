import json,subprocess
import numpy as np
from pathlib import Path
from waveform.coupled import simulate
from waveform.mesh import make_mesh

def test_python_and_js_identical_3d_fsi():
    root=Path(__file__).resolve().parents[1]
    j=json.loads(subprocess.check_output(['node',str(root/'tests/export_engine.cjs')],text=True))
    r=simulate(steps=360,nx=24,ny=13,nz=13,nphi=16,frames=2)
    for key,value,rtol,atol in [
      ('signal',r['signal'],5e-9,5e-9),('wall',r['displacement'].ravel(),1e-8,1e-10),
      ('energy',r['energy'],1e-8,1e-12),('vertices',r['mesh'].rest.ravel(),1e-11,1e-12),
      ('normals',r['mesh'].normals.ravel(),1e-11,1e-12)]:
        np.testing.assert_allclose(j[key],value,rtol=rtol,atol=atol,err_msg=key)
    np.testing.assert_allclose(j['dt'],r['dt'],rtol=1e-12)
    np.testing.assert_allclose(j['max'],r['max_wall_m'],rtol=1e-9)
