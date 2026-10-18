"""Deterministic STL exports from parametric OpenSCAD source (run where OpenSCAD exists)."""
from pathlib import Path
import subprocess,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
CAD=ROOT/'hardware'/'cad'
PARTS=('base','link1','link2','pen')

def build():
    if not shutil.which('openscad'): raise RuntimeError('Install OpenSCAD CLI before exporting STL')
    report={}
    for part in PARTS:
        out=CAD/f'{part}.stl'
        cmd=['openscad','-o',str(out),'-D',f'part="{part}"',str(CAD/'hdr_two_link.scad')]
        res=subprocess.run(cmd,timeout=120,capture_output=True,text=True)
        if res.returncode!=0:raise RuntimeError(f'{part} CAD generation failed: {res.stderr[-1200:]}')
        if out.stat().st_size<1000:raise RuntimeError(f'Empty {part} STL')
        try:import trimesh
        except ImportError: raise RuntimeError('Install trimesh and networkx to validate STL solidity')
        mesh=trimesh.load(out,force='mesh')
        if not(mesh.is_watertight and len(mesh.split(only_watertight=False))==1 and mesh.volume>0):
            raise RuntimeError(f'Nonmanifold or disconnected {part} STL')
        report[part]={'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                     'triangles':len(mesh.faces),'volume_mm3':round(float(mesh.volume),2),
                     'size_mm':[round(float(x),2) for x in mesh.extents]}
    (ROOT/'results'/'cad_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':print(json.dumps(build(),indent=2))
