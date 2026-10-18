"""One-clock mechanical harmonic computer. Inference updates the shaft, not a global solver.

Ideal gears enforce theta_i = k_i*shaft_angle + mount_i. The terminal point is
V0 = c + sum_i radius_i * exp(1j*theta_i), i.e. the linkage resultant.
A hardware linkage computes this geometrically; Python emulates it numerically.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
import numpy as np

TAU = 2 * math.pi

@dataclass(frozen=True)
class Gear:
    turns: int
    x: float
    y: float

    def __post_init__(self):
        if self.turns == 0 or not isinstance(self.turns, int):
            raise ValueError("Nonzero integer gear ratio required")
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError("Finite gear coefficients required")

    @property
    def length(self):
        return math.hypot(self.x, self.y)

    @property
    def phase(self):
        return math.atan2(self.y, self.x)

@dataclass
class Design:
    base_x: float
    base_y: float
    gears: list[Gear]

    def __post_init__(self):
        if not (math.isfinite(self.base_x) and math.isfinite(self.base_y)):
            raise ValueError("Finite base required")
        if len(set(g.turns for g in self.gears)) != len(self.gears):
            raise ValueError("Duplicate signed gear ratios")

    def sample(self, phases):
        p = np.asarray(phases, dtype=np.float64)
        x = np.full(p.shape, self.base_x, dtype=np.float64)
        y = np.full(p.shape, self.base_y, dtype=np.float64)
        for gear in self.gears:
            c = np.cos(gear.turns * p)
            s = np.sin(gear.turns * p)
            x += gear.x*c - gear.y*s
            y += gear.x*s + gear.y*c
        return np.stack((x,y), axis=-1)

    def tip(self, shaft_angle):
        return self.sample(shaft_angle).tolist()

    def to_json(self):
        return {"format":"hdr-gear-ai/1", "base":[self.base_x,self.base_y],
                "gears":[{"k":g.turns,"a":[g.x,g.y]} for g in self.gears]}

    @classmethod
    def from_json(cls, obj):
        if obj.get('format') != 'hdr-gear-ai/1':
            raise ValueError('Unknown machine schema')
        a,b = obj['base']
        return cls(float(a),float(b),[Gear(int(d['k']),float(d['a'][0]),float(d['a'][1])) for d in obj['gears']])

class Mechanism:
    """Stateful finite-torque shaft and ideal gearing; no argmin in the inference loop.

    `command` is the desired shaft ANGLE in unwrapped radians. `step` applies
    velocity servo with finite torque, inertia and viscous drag. For cyclic
    sweeps `drive_speed` supplies angular velocity directly. `read` is a sensor.
    """
    def __init__(self, design: Design, inertia=1.0, servo_kp=14.0, servo_kd=7.0,
                 max_torque=35.0, drag=0.06):
        self.design = design
        self.angle = 0.0
        self.velocity = 0.0
        self.inertia = float(inertia) + sum(.012*(g.turns**2)*(g.length**2) for g in design.gears)
        self.kp = servo_kp
        self.kd = servo_kd
        self.max_torque = max_torque
        self.drag = drag
        self.time = 0.

    def step(self, dt, command=None, drive_speed=None):
        if not 0 < dt <= 0.05: raise ValueError("0 < dt <= 0.05 required")
        if (command is None) == (drive_speed is None): raise ValueError("Choose angle or speed control")
        if command is not None:
            torque = self.kp*(command-self.angle)-self.kd*self.velocity
        else:
            torque = self.kp*(drive_speed-self.velocity)
        torque = float(np.clip(torque,-self.max_torque,self.max_torque))
        self.velocity += dt * (torque-self.drag*self.velocity)/self.inertia
        self.angle += dt*self.velocity
        self.time += dt
        return self.read()

    def read(self):
        return self.design.tip(self.angle)

    def sensed_phases(self):
        return {g.turns:g.turns*self.angle+g.phase for g in self.design.gears}

    def replace_structure(self, new_design):
        """No reset of crank state when gears are added/removed."""
        self.design = new_design
        self.inertia = max(.001, self.inertia)  # Inertia continuity; not an energy-free instant physical retrofit