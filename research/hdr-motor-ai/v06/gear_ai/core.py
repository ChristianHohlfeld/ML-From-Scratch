"""Two-port mechanizable phase computer.

Every rotor has phase k*theta_A+l*theta_B+mount, realizable with signed
gear trains and differential summing. The output is a serial linkage resultant.
The physical realization is a PROPOSAL; numerical emulation is not hardware.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Stage:
    k: int
    l: int
    x: float
    y: float
    def __post_init__(self):
        if type(self.k) is not int or type(self.l) is not int or not (self.k or self.l):
            raise ValueError("Nonzero integer compound ratio required")
        if not all(map(math.isfinite, (self.x, self.y))):
            raise ValueError("Non-finite arm coefficient")
    @property
    def radius(self): return math.hypot(self.x,self.y)
    @property
    def mount(self): return math.atan2(self.y,self.x)
    @property
    def mixed(self): return bool(self.k and self.l)


@dataclass
class Machine:
    origin: complex
    stages: list[Stage]
    def __post_init__(self):
        self.origin=complex(self.origin)
        if not math.isfinite(self.origin.real) or not math.isfinite(self.origin.imag): raise ValueError("Non-finite origin")
        if len({(s.k,s.l) for s in self.stages})!=len(self.stages): raise ValueError("Repeated compound ratios")
    def complex_sample(self, theta_a, theta_b):
        a,b=np.broadcast_arrays(np.asarray(theta_a,dtype=float),np.asarray(theta_b,dtype=float))
        if not (np.isfinite(a).all() and np.isfinite(b).all()): raise ValueError("Non-finite motor input")
        z=np.full(a.shape,self.origin,dtype=complex)
        for s in self.stages:
            z+=(s.x+1j*s.y)*np.exp(1j*(s.k*a+s.l*b))
        return z
    def read(self, a, b):
        z=self.complex_sample(a,b)
        return np.stack((z.real,z.imag),axis=-1)
    def ports(self, a, b):
        return {(s.k,s.l):s.k*a+s.l*b+s.mount for s in self.stages}
    def to_json(self):
        return {'format':'hdr-gear-ai/2','origin':[self.origin.real,self.origin.imag],
                'stages':[{'k':s.k,'l':s.l,'a':[s.x,s.y]} for s in self.stages]}
    @classmethod
    def from_json(cls,obj):
        if obj.get('format')!='hdr-gear-ai/2': raise ValueError('Unknown schema')
        def exact_int(t):
            if type(t) is not int: raise ValueError('Ratio must be integer')
            return t
        return cls(complex(*obj['origin']),[Stage(exact_int(v['k']),exact_int(v['l']),float(v['a'][0]),float(v['a'][1])) for v in obj['stages']])

class Drive:
    """Two independently actuated inertial inputs; no search at readout time.

    Commanded shaft angles may change independently. Gear relation is k*A+l*B.
    A proposed mechanical differential could implement this relation; here it is
    numerically evaluated exactly. Servo applies bounded torque and damping.
    """
    def __init__(self,machine:Machine,inertia=(1,1),kp=24.,kd=9.,max_torque=40.,drag=.10):
        self.machine=machine
        self.angle=np.zeros(2)
        self.speed=np.zeros(2)
        self.inertia=np.array(inertia,dtype=float)
        if self.inertia.shape!=(2,) or np.any(self.inertia<=0):raise ValueError('Positive inertia per shaft')
        self.kp,self.kd,self.max_torque,self.drag=kp,kd,max_torque,drag
        self._refresh_inertia()
        self.time=0.
    def step(self,dt,targets):
        if not 0<dt<=.025: raise ValueError('0<dt<=.025')
        target=np.asarray(targets,dtype=float)
        if target.shape!=(2,) or not np.isfinite(target).all():raise ValueError('Two finite targets required')
        torque=np.clip(self.kp*(target-self.angle)-self.kd*self.speed,-self.max_torque,self.max_torque)
        # Lagrange reduction of constrained rotors: mixed gears back-drive both
        # inputs through the off-diagonal entries of the inertia matrix.
        accel=np.linalg.solve(self.mass_matrix,torque-self.drag*self.speed)
        self.speed+=dt*accel
        self.angle+=dt*self.speed
        self.time+=dt
        return self.read()
    def read(self): return self.machine.read(*self.angle).tolist()
    def _refresh_inertia(self):
        M=np.diag(self.inertia.copy())
        for s in self.machine.stages:
            r=np.array([s.k,s.l],dtype=float)
            stage_inertia=.02*(s.radius**2)
            M+=stage_inertia*np.outer(r,r)
        self.mass_matrix=M
    def replace(self,machine):
        self.machine=machine
        self._refresh_inertia() # topology changes only during offline structure search
        # Input angles/velocities persist; changing real rotors under load
        # would require stopping the device and accounting for energy transfer.


def rectangular_interaction(machine,a0,a1,b0,b1):
    """For any additive-only machine this mixed finite difference is zero."""
    p=lambda a,b:machine.complex_sample(a,b)
    return complex(p(a1,b1)-p(a1,b0)-p(a0,b1)+p(a0,b0))
