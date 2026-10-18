"""Export closed 3D triangular tube surfaces for external inspection.

This creates a solid envelope with planar end caps (not a CAD drawing of
human anatomy). Does not export the internal rod hypothesis.
"""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def tube_obj(path, shape, rings=70, sides=32):
    from waveform import geometry
    mask,dx,info=geometry(shape,nx=rings,ny=19,nz=19)
    vx=[]; faces=[]
    for i in range(rings):
        x=i*dx
        for j in range(sides):
            phi=2*np.pi*j/sides
            vx.append((x,float(info['ry'][i]*np.cos(phi)),float(info['rz'][i]*np.sin(phi))))
    # Ring mesh: consistent CCW normals viewed from outside
    for i in range(rings-1):
        for j in range(sides):
            j1=(j+1)%sides
            a=i*sides+j; b=(i+1)*sides+j
            c=(i+1)*sides+j1; d=i*sides+j1
            faces.extend([(a,c,b),(a,d,c)])
    beg=len(vx);vx.append((0.,0.,0.));end=len(vx);vx.append(((rings-1)*dx,0.,0.))
    for j in range(sides):
        j1=(j+1)%sides
        faces.append((beg,j1,j))
        faces.append((end,(rings-1)*sides+j,(rings-1)*sides+j1))
    s=['# HDR WaveForm synthetic shape; metres']
    for x,y,z in vx:s.append(f'v {x:.8f} {y:.8f} {z:.8f}')
    for a,b,c in faces:s.append(f'f {a+1} {b+1} {c+1}')
    path.write_text('\n'.join(s)+'\n')
    return len(vx),len(faces)
if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    from waveform import Shape
    d=json.loads((ROOT/'results/benchmark.json').read_text())['cases'][0]
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for name,ctrl in [('reference',d['truth']),('learned',d['recovered'])]:
        fn=out/f'waveform_{name}.obj'
        print(name,*tube_obj(fn,Shape(tuple(ctrl))))
