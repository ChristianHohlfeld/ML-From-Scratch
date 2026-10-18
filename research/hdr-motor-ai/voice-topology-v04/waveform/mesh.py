"""3-D triangulated tubular membrane with independently moving XYZ vertices.

The material manifold is a 2D (axial × circumferential) closed triangular
surface embedded in 3D. End rings are clamped to the frame. Coordinates are
physical metres; topology is unchanged during the small-deformation solver.
"""
from dataclasses import dataclass
import numpy as np
from .core import Shape, geometry

@dataclass(frozen=True)
class Mesh:
    rest: np.ndarray            # nx, nphi, 3
    normals: np.ndarray         # nx, nphi, 3
    patch_area: np.ndarray      # nx, nphi, m²
    cell: np.ndarray            # nx, nphi, voxel flat index
    triangles: np.ndarray       # [(a,b,c)] indexed vertices
    edges: np.ndarray           # (a,b) surface mesh edges
    dx: float
    dphi: float


def make_mesh(shape:Shape=Shape(),*,nx=28,ny=15,nz=15,nphi=16,extent=.084):
    if nphi<8 or nphi%2:raise ValueError('circumference requires even nphi >= 8')
    mask, dx, info=geometry(shape,nx=nx,ny=ny,nz=nz,extent=extent)
    dy,dz=info['dy'],info['dz']; ph=np.arange(nphi)*2*np.pi/nphi
    rest=np.empty((nx,nphi,3)); normals=np.empty_like(rest); patch=np.empty((nx,nphi)); cells=np.empty((nx,nphi),dtype=np.int64)
    triangles=[]; edges=set()
    for x in range(nx):
        for j,t in enumerate(ph):
            co,si=np.cos(t),np.sin(t); ry,rz=info['ry'][x],info['rz'][x]
            envelope=np.sin(np.pi*x/(nx-1))
            rr=1.+envelope*(shape.lobes[0]*np.cos(2*t)+shape.lobes[1]*np.sin(3*t))
            dr=envelope*(-2*shape.lobes[0]*np.sin(2*t)+3*shape.lobes[1]*np.cos(3*t))
            y=ry*rr*co;z=rz*rr*si
            yp=ry*(dr*co-rr*si);zp=rz*(dr*si+rr*co)
            rest[x,j]=[x*dx,y,z]
            normal=np.array([0.,zp,-yp]);normal/=np.linalg.norm(normal)
            normals[x,j]=normal
            patch[x,j]=dx*(2*np.pi/nphi)*np.hypot(yp,zp)
            # sample pressure from an interior voxel, never from a geometric formula
            for factor in (.90,.80,.65,.50,.0):
                iy=int(np.clip(round((factor*y+extent/2)/dy),0,ny-1))
                iz=int(np.clip(round((factor*z+extent/2)/dz),0,nz-1))
                if mask[x,iy,iz]:break
            if not mask[x,iy,iz]:raise ValueError('no interior voxel at surface')
            cells[x,j]=(x*ny+iy)*nz+iz
    for x in range(nx-1):
        for j in range(nphi):
            a=x*nphi+j;b=x*nphi+(j+1)%nphi;c=(x+1)*nphi+j;d=(x+1)*nphi+(j+1)%nphi
            triangles.extend([(a,c,b),(b,c,d)])
            for v,w in ((a,b),(a,c),(b,c),(c,d),(b,d)):
                edges.add((min(v,w),max(v,w)))
    return Mesh(rest,normals,patch,cells,np.asarray(triangles,dtype=np.int32),np.asarray(sorted(edges),dtype=np.int32),dx,2*np.pi/nphi)


def deformed_vertices(mesh:Mesh, displacement:np.ndarray):
    if displacement.shape!=mesh.rest.shape:raise ValueError('displacement shape differs')
    return mesh.rest+displacement


def export_obj(mesh:Mesh, displacement=None):
    vertices=mesh.rest if displacement is None else deformed_vertices(mesh,displacement)
    lines=['# HDR WaveForm v0.4 triangulated 3D membrane (end caps intentionally open)']
    for x,y,z in vertices.reshape(-1,3):lines.append(f'v {x:.9g} {y:.9g} {z:.9g}')
    for a,b,c in mesh.triangles:lines.append(f'f {a+1} {b+1} {c+1}')
    return '\n'.join(lines)+'\n'
