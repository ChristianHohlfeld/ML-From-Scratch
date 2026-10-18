"""Write deterministic triangulated surface specimens to interactive lab."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark import CLASSES, base_mesh,equalize_vertices,topology
out={}
for i,k in enumerate(CLASSES):
 m=equalize_vertices(base_mesh(k),96,121+i)
 out[k]={'vertices':m.vertices.round(6).tolist(),'faces':m.faces.tolist(), 'edges':m.edges_unique.tolist(), 'topology':topology(m)}
p=Path(__file__).resolve().parent.parent/'web'/'meshes.js'
p.write_text('const HDR_MESHES = '+json.dumps(out,separators=(',',':'))+';\n')
print(p,p.stat().st_size)
