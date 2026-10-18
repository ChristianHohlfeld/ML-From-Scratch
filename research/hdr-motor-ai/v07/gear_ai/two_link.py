"""2R planar serial linkage: a physically realizable coupled machine.

Motor A actuates shoulder angle alpha. Motor B is carried on arm 1 and
actuates elbow angle beta relative to arm 1. Both are independently driven;
the physical pen tip represents c + L1 exp(i alpha) + L2 exp(i(alpha+beta)).
In hardware the output is *observed optically*, never computed by firmware.

Digital twin is deliberately approximate: horizontal plane, point-mass elbow
motor, uniform slender links, point mass at tip, and torque-limited actuators.
No implicit solver or argmin runs at inference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class TwoLink:
    l1: float = 0.060  # metres, centre of shoulder to elbow
    l2: float = 0.045  # metres, centre of elbow to visible tip
    m1: float = 0.012  # kg, arm 1 effective mass
    m2: float = 0.008  # kg, arm 2 effective mass
    motor_b: float = 0.009  # kg, SG90 mass at elbow
    tip_mass: float = 0.003  # kg, red optical target / pen
    rotor_inertia: tuple[float, float] = (3e-6, 3e-6)
    def __post_init__(self):
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.l1, self.l2, self.m1, self.m2, self.motor_b,
                self.tip_mass, *self.rotor_inertia)):
            raise ValueError('Positive finite physical parameters required')
    def tip(self, a: float, b: float) -> np.ndarray:
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError('Finite joint angles required')
        return np.array([self.l1*math.cos(a)+self.l2*math.cos(a+b),
                         self.l1*math.sin(a)+self.l2*math.sin(a+b)])
    def jacobian(self, a: float, b: float) -> np.ndarray:
        return np.array([[-self.l1*math.sin(a)-self.l2*math.sin(a+b), -self.l2*math.sin(a+b)],
                         [self.l1*math.cos(a)+self.l2*math.cos(a+b), self.l2*math.cos(a+b)]])
    def mass_matrix(self, b: float) -> np.ndarray:
        l1,l2,m1,m2,mb,mt=self.l1,self.l2,self.m1,self.m2,self.motor_b,self.tip_mass
        cs=math.cos(b)
        p=m2*l1*l2/2+mt*l1*l2
        m11=(m1/3+mb+m2+mt)*l1*l1+(m2/3+mt)*l2*l2+2*p*cs+self.rotor_inertia[0]
        m12=(m2/3+mt)*l2*l2+p*cs
        m22=(m2/3+mt)*l2*l2+self.rotor_inertia[1]
        return np.array([[m11,m12],[m12,m22]])
    def coriolis(self, b: float, da: float, db: float) -> np.ndarray:
        h=(self.m2/2+self.tip_mass)*self.l1*self.l2*math.sin(b)
        return np.array([-h*(2*da*db+db*db),h*da*da])
    def rectangular_interaction(self,a0,a1,b0,b1) -> np.ndarray:
        return self.tip(a1,b1)-self.tip(a1,b0)-self.tip(a0,b1)+self.tip(a0,b0)


@dataclass
class MotorState:
    angles: np.ndarray = field(default_factory=lambda:np.zeros(2))
    speeds: np.ndarray = field(default_factory=lambda:np.zeros(2))
    seconds: float = 0.0


class PhysicalTwin:
    """Finite-torque simulation, *not* SG90 firmware or real-measured speed.

    Independent motor angle targets, joint coupling from inertial terms.
    The firmware commands servo positions; actual servo response is unknown.
    """
    def __init__(self, geom:TwoLink|None=None, kp=.08, kd=.012, cap=.045,
                 friction=.0015, limit_deg=55.):
        self.geom=geom or TwoLink()
        self.state=MotorState()
        self.kp,self.kd,self.cap,self.friction=kp,kd,cap,friction
        self.limit=math.radians(limit_deg)
        if min(kp,kd,cap,friction)>=0 and cap>0 and 0<limit_deg<85: pass
        else: raise ValueError('Invalid drive parameters')
    def step(self, dt:float, target:tuple[float,float]):
        """Angles in radians. Semi-implicit Euler; fixed timestep <= 2 ms."""
        if not (0<dt<=.002): raise ValueError('0 < dt <= 0.002')
        tgt=np.asarray(target,dtype=float)
        if tgt.shape!=(2,) or not np.isfinite(tgt).all() or np.any(np.abs(tgt)>self.limit):
            raise ValueError('Both target angles must be finite and within soft limits')
        s=self.state
        tau=np.clip(self.kp*(tgt-s.angles)-self.kd*s.speeds,-self.cap,self.cap)
        rhs=tau-self.geom.coriolis(s.angles[1],*s.speeds)-self.friction*s.speeds
        acc=np.linalg.solve(self.geom.mass_matrix(s.angles[1]),rhs)
        s.speeds+=acc*dt
        s.angles+=s.speeds*dt
        s.seconds+=dt
        return self.geom.tip(*s.angles)
    def settle(self,target,seconds=.35,dt=.001):
        steps=round(seconds/dt)
        for _ in range(steps):self.step(dt,target)
        return self.geom.tip(*self.state.angles)
