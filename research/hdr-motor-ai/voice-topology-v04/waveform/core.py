"""3D scalar pressure-wave FDTD in a discretised, deformable acoustic tube.

Deliberately simple research solver, *not* full Navier-Stokes, not Box3D and
not a physically validated human vocal-tract model. Acoustic pressure lives
on voxels; solid walls impose a nearest-neighbour no-flux approximation.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

C = 343.0  # metres/second


@dataclass(frozen=True)
class Shape:
    """Dimensionless normalized width controls; topology is an explicit choice.

    controls scale cross-sectional semi-axes, not absolute millimetres.
    """
    controls: tuple[float, ...] = (0.88, 1.15, 0.72, 1.08)
    length: float = 0.168
    aspect: float = 0.88
    rod: bool = False
    lobes: tuple[float,float] = (0.0,0.0)  # circumferential rest-shape modes

    def __post_init__(self):
        if len(self.controls) != 4 or any(not .50 <= v <= 1.48 for v in self.controls):
            raise ValueError('four radial controls must each lie within [0.5,1.48]')
        if len(self.lobes)!=2 or any(not -.25 <= v <= .25 for v in self.lobes):
            raise ValueError('angular form coefficients outside [-.25,.25]')
        if not .12 <= self.length <= .23 or not .55 <= self.aspect <= 1.35:
            raise ValueError('outside supported physical geometry range')


def geometry(shape: Shape, *, nx=38, ny=19, nz=19, extent=0.084):
    """Boolean 3D fluid volume and geometric area-profile, in SI units.

    The tube is aligned with x; shape is a *genuinely spatial* deformable
    elliptic tube, not a 1D digital filter or a phase-space embedding.
    """
    if min(nx, ny, nz) < 8: raise ValueError('grid too small')
    dx = shape.length / (nx - 1)
    y = np.linspace(-extent/2, extent/2, ny)
    z = np.linspace(-extent/2, extent/2, nz)
    u = np.linspace(0, 1, nx)
    centres = np.linspace(0, 1, 6)
    values = np.r_[1., np.asarray(shape.controls), 1.]
    radial = np.interp(u, centres, values)
    # Lowpass the piecewise-linear shape slightly, keeping endpoints fixed.
    radial = np.convolve(np.pad(radial, (2,2), mode='edge'),
                         np.array([1,4,6,4,1])/16, mode='valid')
    base = .019
    ry = base * radial
    rz = ry * shape.aspect
    yy, zz = y[None,:,None], z[None,None,:]
    YY=yy/ry[:,None,None];ZZ=zz/rz[:,None,None]
    norm=np.sqrt(YY**2+ZZ**2)
    theta=np.arctan2(ZZ,YY)
    envelope=np.sin(np.pi*u)[:,None,None]
    angular=1.+envelope*(shape.lobes[0]*np.cos(2*theta)+shape.lobes[1]*np.sin(3*theta))
    mask = norm <= angular
    if shape.rod:
        # Continuous axial obstacle. Annular cross-sections have an H1 loop.
        r_inner = .0055
        mask &= (yy**2 + zz**2 >= r_inner**2)
    if not mask.any(): raise ValueError('empty domain')
    return mask, dx, {'ry':ry,'rz':rz,'area':np.pi*ry*rz,'radial':radial,
                      'dy':float(y[1]-y[0]),'dz':float(z[1]-z[0])}


@njit(cache=True)
def _integrate(mask, drive, source_plane, probe_plane, q, steps, lamx, lamy, lamz, damping):
    nx,ny,nz=mask.shape
    prev=np.zeros((nx,ny,nz), dtype=np.float64)
    p=np.zeros_like(prev)
    nxt=np.zeros_like(prev)
    probe=np.zeros(steps, dtype=np.float64)
    ep=np.zeros(steps, dtype=np.float64)
    for t in range(steps):
        for x in range(1,nx-1):
            for y in range(1,ny-1):
                for z in range(1,nz-1):
                    if not mask[x,y,z]: continue
                    v=p[x,y,z]
                    lap=0.0
                    if mask[x+1,y,z]: lap+=lamx*(p[x+1,y,z]-v)
                    if mask[x-1,y,z]: lap+=lamx*(p[x-1,y,z]-v)
                    if mask[x,y+1,z]: lap+=lamy*(p[x,y+1,z]-v)
                    if mask[x,y-1,z]: lap+=lamy*(p[x,y-1,z]-v)
                    if mask[x,y,z+1]: lap+=lamz*(p[x,y,z+1]-v)
                    if mask[x,y,z-1]: lap+=lamz*(p[x,y,z-1]-v)
                    # small interior dissipation + stronger absorbing regions
                    decay = damping
                    if x < 4 or x > nx - 5: decay *= 9.0
                    value=(2.-decay)*v-(1.-decay)*prev[x,y,z]+lap
                    if x == source_plane: value += drive[t] * q
                    nxt[x,y,z]=value
        acc=0.0; n=0
        for y in range(1,ny-1):
            for z in range(1,nz-1):
                if mask[probe_plane,y,z]:
                    acc+=nxt[probe_plane,y,z]; n+=1
        probe[t]=acc/max(n,1)
        # average instantaneous pressure energy (sampling diagnostics)
        ee=0.
        for y in range(1,ny-1):
            for z in range(1,nz-1):
                if mask[probe_plane,y,z]:ee+=nxt[probe_plane,y,z]**2
        ep[t]=ee/max(n,1)
        tmp=prev; prev=p; p=nxt; nxt=tmp
    return probe,ep


def excite(steps, dt, *, variant='chirp'):
    t=np.arange(steps)*dt
    # Smooth onset and offset. Chirp is multi-frequency and deterministic.
    fade=np.sin(np.pi*np.minimum(np.arange(steps)/max(steps-1,1),1))**2
    if variant=='chirp':
        f0,f1=330.,3900.
        phase=2*np.pi*(f0*t + .5*(f1-f0)/max(t[-1],dt)*t*t)
        return fade*np.sin(phase)
    if variant=='multitone':
        return fade*(np.sin(2*np.pi*470*t)+.6*np.sin(2*np.pi*1160*t+.4)+.45*np.sin(2*np.pi*2520*t+.8))
    if variant=='impulse':
        n=np.arange(steps)
        return np.exp(-.5*((n-steps//14)/10)**2)
    raise ValueError(variant)


def propagate(shape: Shape, *, steps=2100, nx=38, ny=19, nz=19,
              excitation='chirp', return_field=False):
    """Advance actual 3D wave field, not a 1D filter.

    CFL: 3*lambda² <= 1, lambda = c*dt/dx. Fixed dt obeys
    the 3D stability limit; solid walls are no-flux voxel boundaries.
    """
    mask,dx,info=geometry(shape,nx=nx,ny=ny,nz=nz)
    dy,dz=info['dy'],info['dz']
    dt=.78/(C*np.sqrt(1/dx**2+1/dy**2+1/dz**2))
    lamx,lamy,lamz=(C*dt/dx)**2,(C*dt/dy)**2,(C*dt/dz)**2
    assert lamx+lamy+lamz<1
    signal=excite(steps,dt,variant=excitation)
    source=2; probe=nx-4
    pressure,energy=_integrate(mask,signal,source,probe,.035,steps,lamx,lamy,lamz,.003)
    return {'signal':pressure, 'drive':signal,'energy':energy,'dt':dt,'dx':dx,
            'mask':mask if return_field else None,
            'area':info['area'], 'radial':info['radial'],
            'cfl':np.sqrt(lamx+lamy+lamz),'shape':shape}


def signature(record, *, bins=32, low=200, high=3900):
    """Normalized complex-input/output magnitude transfer spectrum.

    Ignore nearly unexcited bins to avoid numerical noise; interpolate only
    within reliably driven frequencies. Shape is acoustically distinguishable
    only to extent measured by this limited-band signature.
    """
    y=record['signal']; x=record['drive']; dt=record['dt']
    nfft=max(8192,2**int(np.ceil(np.log2(len(x)*2))))
    fy=np.fft.rfft(y*np.hanning(len(y)),n=nfft)
    fx=np.fft.rfft(x*np.hanning(len(x)),n=nfft)
    f=np.fft.rfftfreq(nfft,dt)
    f_query=np.geomspace(low,high,bins)
    power=np.abs(fx)
    # Regularized estimate of spectral response; no divisions by near-zero.
    ratio=np.abs(fy)*power/(power**2 + (power.max()*.045)**2)
    vec=np.interp(f_query,f,ratio)
    vec=np.log1p(50*vec)
    vec=vec/(np.linalg.norm(vec)+1e-12)
    return vec


def topology_2d(shape:Shape):
    """Analytic Betti numbers for the intended tube fluid cross-section.

    This is a property of the *boundary topology*, NOT an inference from sound.
    A continuous central rod gives one annular hole. Does not measure H2.
    """
    return (1, 1 if shape.rod else 0)


def similarity(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.linalg.norm(a-b))
