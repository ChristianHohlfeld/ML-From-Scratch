"""Bidirectionally coupled 3D acoustics and a triangulated 3D soft membrane.

Each of nx*nphi surface vertices has independent XYZ displacement and velocity.
Nodes are joined by axial/circumferential membrane springs and respond to local
pressure. The same surface-patch flux drives pressure in the corresponding fluid
voxel: force p*A*n and flux (A/V)*(v·n) are reciprocal work terms.

Linearized *fixed Eulerian fluid voxel mask*: correct to first order in small
wall motion; no remeshing, finite-strain continuum elasticity, air viscosity,
or contact solver. It is an explicit numerical fluid-structure coupling, not
Box3D or a simulated full physiological vocal tract.
"""
from dataclasses import dataclass, replace
import numpy as np
from numba import njit
from .core import Shape, geometry, excite, signature
from .mesh import make_mesh

RHO=1.2; C=343.; BULK=RHO*C*C

@dataclass(frozen=True)
class Wall:
    enabled:bool=True
    mass_per_area:float=.14
    stiffness_per_area:float=2.1e6
    damping_per_area:float=34.
    tension:float=3.
    acoustic_loss:float=8.
    source_gain:float=.35
    def __post_init__(self):
        if self.mass_per_area<=0 or self.stiffness_per_area<=0 or min(self.damping_per_area,self.tension,self.acoustic_loss,self.source_gain)<0:
            raise ValueError('passive mass/stiffness and nonnegative damping required')

@njit(cache=True)
def _run(mask,drive,dt,dx,dy,dz,cell,normal,patch,params,enabled,frames):
    nx,ny,nz=mask.shape; npf=cell.shape[1]; n=len(drive)
    p=np.zeros((nx,ny,nz));vx=np.zeros((nx-1,ny,nz));vy=np.zeros((nx,ny-1,nz));vz=np.zeros((nx,ny,nz-1))
    u=np.zeros((nx,npf,3));v=np.zeros((nx,npf,3)); flux=np.zeros((nx*ny*nz));
    sig=np.zeros(n); energy=np.zeros(n); trace=np.zeros(n); local=np.zeros(n)
    psnap=np.zeros((frames,nx,ny),dtype=np.float32)
    usnap=np.zeros((frames,nx,npf,3),dtype=np.float32)
    m,k,b,T,loss,gain=params;vol=dx*dy*dz
    fidx=0
    for t in range(n):
        for x in range(nx-1):
            for y in range(ny):
                for z in range(nz):
                    vx[x,y,z]=vx[x,y,z]-dt/(RHO*dx)*(p[x+1,y,z]-p[x,y,z]) if mask[x,y,z] and mask[x+1,y,z] else 0.
        for x in range(nx):
            for y in range(ny-1):
                for z in range(nz):
                    vy[x,y,z]=vy[x,y,z]-dt/(RHO*dy)*(p[x,y+1,z]-p[x,y,z]) if mask[x,y,z] and mask[x,y+1,z] else 0.
            for y in range(ny):
                for z in range(nz-1):
                    vz[x,y,z]=vz[x,y,z]-dt/(RHO*dz)*(p[x,y,z+1]-p[x,y,z]) if mask[x,y,z] and mask[x,y,z+1] else 0.
        if enabled:
            for q in range(len(flux)):flux[q]=0.
            old=u.copy()
            for x in range(1,nx-1):
                for j in range(npf):
                    jj=(j+1)%npf;jm=(j-1+npf)%npf
                    q=cell[x,j];xx=q//(ny*nz);yy=q//nz%ny;zz=q%nz
                    # Vector-valued 3-D membrane displacement; normal pressure load.
                    # Tension is a Laplacian with physical axial and angular lengths.
                    circum_len=max(1.e-5,patch[x,j]/dx)
                    circum_scale=(2*np.pi/npf/circum_len)**2
                    for dim in range(3):
                        axial=(old[x-1,j,dim]-2*old[x,j,dim]+old[x+1,j,dim])/(dx*dx)
                        angular=(old[x,jm,dim]-2*old[x,j,dim]+old[x,jj,dim])*circum_scale
                        accel=(normal[x,j,dim]*p[xx,yy,zz]-k*old[x,j,dim]-b*v[x,j,dim]+T*(axial+angular))/m
                        v[x,j,dim]+=dt*accel
            for x in range(1,nx-1):
                for j in range(npf):
                    dot=0.
                    for d in range(3):
                        u[x,j,d]+=dt*v[x,j,d]
                        dot+=v[x,j,d]*normal[x,j,d]
                    flux[cell[x,j]]+=patch[x,j]/vol*dot
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if not mask[x,y,z]:continue
                    div=0.
                    if x<nx-1:div+=vx[x,y,z]/dx
                    if x>0:div-=vx[x-1,y,z]/dx
                    if y<ny-1:div+=vy[x,y,z]/dy
                    if y>0:div-=vy[x,y-1,z]/dy
                    if z<nz-1:div+=vz[x,y,z]/dz
                    if z>0:div-=vz[x,y,z-1]/dz
                    p[x,y,z]=(1-loss*dt)*p[x,y,z]-BULK*dt*(div+flux[(x*ny+y)*nz+z] if enabled else div)
                    if x==2:p[x,y,z]+=gain*drive[t]
        vout=0;num=0
        for y in range(ny):
            for z in range(nz):
                if mask[nx-4,y,z]:vout+=p[nx-4,y,z];num+=1
        sig[t]=vout/max(num,1)
        mid=nx//2
        rad=0.;asym=0.
        for j in range(npf):
            for d in range(3):rad+=u[mid,j,d]*normal[mid,j,d]
            if j==0:asym+=u[mid,j,1]
            if j==npf//2:asym-=u[mid,j,1]
        trace[t]=rad/npf;local[t]=asym
        e=0.
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if mask[x,y,z]:e+=.5*p[x,y,z]**2/BULK*vol
        for x in range(nx-1):
            for y in range(ny):
                for z in range(nz):e+=.5*RHO*vx[x,y,z]**2*vol
        for x in range(nx):
            for y in range(ny-1):
                for z in range(nz):e+=.5*RHO*vy[x,y,z]**2*vol
            for y in range(ny):
                for z in range(nz-1):e+=.5*RHO*vz[x,y,z]**2*vol
        if enabled:
            for x in range(1,nx-1):
                for j in range(npf):
                    jm=(j-1+npf)%npf
                    dl=max(1.e-5,patch[x,j]/dx)
                    for d in range(3):
                        e+=.5*patch[x,j]*(m*v[x,j,d]**2+k*u[x,j,d]**2)
                        e+=.5*T*patch[x,j]*((u[x,j,d]-u[x-1,j,d])**2/dx**2+(u[x,j,d]-u[x,jm,d])**2/dl**2)
        energy[t]=e
        if fidx<frames and (t==n-1 or t>=int((fidx+1)*n/(frames+1))-1):
            for x in range(nx):
                for j in range(npf):
                    for d in range(3):usnap[fidx,x,j,d]=u[x,j,d]
                for y in range(ny):psnap[fidx,x,y]=p[x,y,nz//2]
            fidx+=1
    return sig,trace,local,energy,psnap,usnap,u,v


def simulate(shape:Shape=Shape(), *,wall=Wall(),steps=1300,nx=28,ny=15,nz=15,nphi=16,excitation='chirp',drive=None,frames=12):
    if steps<2 or frames<1 or nx<9:raise ValueError('invalid resolution')
    mesh=make_mesh(shape,nx=nx,ny=ny,nz=nz,nphi=nphi)
    mask,dx,info=geometry(shape,nx=nx,ny=ny,nz=nz)
    dy,dz=info['dy'],info['dz']
    dt=.40/(C*np.sqrt(dx**-2+dy**-2+dz**-2))
    spacing=min(float(mesh.patch_area.min()/dx),dx)
    omega=np.sqrt(wall.stiffness_per_area/wall.mass_per_area + 8*wall.tension/(wall.mass_per_area*spacing**2))
    if omega*dt>.23:raise ValueError('surface explicit stability condition violated')
    signal=excite(steps,dt,variant=excitation) if drive is None else np.asarray(drive,dtype=np.float64)
    if signal.shape!=(steps,) or not np.all(np.isfinite(signal)):raise ValueError('invalid source array')
    prm=np.array([wall.mass_per_area,wall.stiffness_per_area,wall.damping_per_area,wall.tension,wall.acoustic_loss,wall.source_gain],float)
    mic,trace,local,energy,psnap,usnap,u,v=_run(mask,signal,dt,dx,dy,dz,mesh.cell,mesh.normals,mesh.patch_area,prm,wall.enabled,frames)
    if not (np.isfinite(mic).all() and np.isfinite(energy).all() and np.isfinite(u).all()):raise RuntimeError('nonfinite FSI')
    maxu=float(np.max(np.linalg.norm(u,axis=2)))
    # Never silently turn linearized voxel FSI into a finite-deformation claim.
    if maxu>min(dx,dy,dz)*.07:raise RuntimeError('membrane moved outside fixed-voxel linear range')
    return {'signal':mic,'drive':signal,'wall_trace':trace,'asymmetry_trace':local,'energy':energy,'snapshots':psnap,'wall_snapshots':usnap,'displacement':u,'velocity':v,'dt':dt,'cfl':C*dt*np.sqrt(dx**-2+dy**-2+dz**-2),'mask':mask,'mesh':mesh,'shape':shape,'wall':wall,'max_wall_m':maxu}


def fit_shape(target,*,wall=Wall(),initial=None,steps=950,max_evals=35,nx=24,ny=13,nz=13,nphi=12,excitation='chirp'):
    """Search *rest shape* through repeated FSI simulations, no synthetic target leakage."""
    from scipy.optimize import minimize
    if initial is None:initial=Shape(controls=(1,1,1,1))
    best=[float('inf'),initial.controls,initial.lobes]; calls=0
    def obj(params):
        nonlocal calls
        shape=replace(initial,controls=tuple(np.clip(params[:4],.5,1.48)),lobes=tuple(np.clip(params[4:],-.25,.25)))
        try:r=simulate(shape,wall=wall,steps=steps,nx=nx,ny=ny,nz=nz,nphi=nphi,excitation=excitation,frames=1)
        except (ValueError,RuntimeError):return 1e4
        err=float(np.mean((signature(r)-target)**2))
        calls+=1
        if err<best[0]:best[:]=[err,shape.controls,shape.lobes]
        return err
    before=obj(np.r_[initial.controls,initial.lobes])
    minimize(obj,np.r_[initial.controls,initial.lobes],method='Powell',bounds=[(.5,1.48)]*4+[(-.25,.25)]*2,options={'maxfev':max_evals,'xtol':.08,'ftol':.0003})
    return {'before':before,'after':best[0],'shape':replace(initial,controls=best[1],lobes=best[2]),'evals':calls}
