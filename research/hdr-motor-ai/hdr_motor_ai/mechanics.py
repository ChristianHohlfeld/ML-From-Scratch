"""Physically grounded *simulated* analog computing: input motors drive passive V0.

Zero algebraic minimization; V0 is an undriven shaft whose angle responds over time
through elastic gear ratios and viscous damping. Learned edge parameters represent
adjustable physical couplers; learning itself is offline and digital (backpropagation).
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MotorState:
    angle: torch.Tensor
    velocity: torch.Tensor


def topology(n: int, groups: int = 2):
    if n < 2 or groups < 1 or groups > n:
        raise ValueError('Requires 2+ motors and 1 <= groups <= motors')
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            edges.append((a, b, 'pair', 1.0))
    for a in range(n - 1):
        edges.append((a, a + 1, 'neighbor', 1.0))
    for a in range(n):
        edges.append((a, n + a % groups, 'group', 1.0))
    for g in range(groups):
        edges.append((n + g, n + groups, 'hub', (-1.0 if g % 2 else 1.0)))
    return edges


class GearNetwork(nn.Module):
    """Differentiable passive gear-spring network. Energies are bounded below.

    Elastic link extension e_ab = angle[a] + ratio_ab * angle[b],
    energy U_ab = k_ab * e_ab^2/2 + cubic_ab * e_ab^4/4.
    Torque = -gradient(U), and driven motors use positive spring to target.
    V0 is never actuated. Output is sensed ONLY from passive V0 channels.
    """
    def __init__(self, n: int = 6, groups: int = 2, channels: int = 3,
                 dt: float = 0.035, damping: float = 2.4, inertia: float = 1.0,
                 spring: float = 0.3, cubic: float = 0.04,
                 motor_stiffness: float = 2.2, ratio_span: float = 0.45):
        super().__init__()
        if channels < 1 or dt <= 0 or inertia <= 0 or damping <= 0:
            raise ValueError('Invalid physical configuration')
        self.n, self.groups, self.channels = n, groups, channels
        self.total, self.v0_idx = n + groups + 1, n + groups
        self.dt, self.inertia, self.ratio_span = dt, inertia, ratio_span
        self.damping, self.motor_stiffness = damping, motor_stiffness
        links = topology(n, groups)
        self.edge_kinds = [kind for _, _, kind, _ in links]
        self.edges = [(a, b, kind, sign) for a, b, kind, sign in links]
        self.register_buffer('a', torch.tensor([e[0] for e in links], dtype=torch.long))
        self.register_buffer('b', torch.tensor([e[1] for e in links], dtype=torch.long))
        self.register_buffer('orientation', torch.tensor([e[3] for e in links], dtype=torch.float32))
        n_edges = len(links)
        self.log_k = nn.Parameter(torch.full((n_edges, channels), float(spring)).log())
        self.log_cubic = nn.Parameter(torch.full((n_edges, channels), float(cubic)).log())
        self.ratio_logits = nn.Parameter(torch.zeros(n_edges, channels))
        self.motor_gain = nn.Parameter(torch.ones(n, channels))
        self.motor_bias = nn.Parameter(torch.zeros(n, channels))
        self.readout = nn.Linear(channels, 1)

    def coefficients(self):
        # Stiffness, positive quartic stiffness and bounded gear ratios.
        k = self.log_k.clamp(-8, 2.3).exp()
        cubic = self.log_cubic.clamp(-10, -0.5).exp()
        ratio = self.orientation[:, None] * (1.0 + self.ratio_span * torch.tanh(self.ratio_logits))
        return k, cubic, ratio

    def make_state(self, u: torch.Tensor):
        zeros = u.new_zeros((u.shape[0], self.total, self.channels))
        return MotorState(zeros, torch.zeros_like(zeros))

    def _validate(self, u, state):
        if u.ndim != 2 or u.shape[1] != self.n:
            raise ValueError(f'Expected motor targets of shape [batch,{self.n}]')
        expected = (u.shape[0], self.total, self.channels)
        if state.angle.shape != expected or state.velocity.shape != expected:
            raise ValueError(f'Expected state {expected}')

    def energy(self, state: MotorState, u: torch.Tensor):
        self._validate(u, state)
        k, cubic, ratio = self.coefficients()
        extension = state.angle[:, self.a, :] + ratio[None] * state.angle[:, self.b, :]
        target = u[:, :, None] * self.motor_gain[None] + self.motor_bias[None]
        kinetic = (0.5 * self.inertia * state.velocity.square()).sum((1, 2))
        potential = (0.5 * k[None] * extension.square() +
                     0.25 * cubic[None] * extension.pow(4)).sum((1, 2))
        potential = potential + 0.5 * self.motor_stiffness * (
            target - state.angle[:, :self.n, :]).square().sum((1, 2))
        return kinetic + potential

    def derivatives(self, state: MotorState, u: torch.Tensor):
        self._validate(u, state)
        k, cubic, ratio = self.coefficients()
        e = state.angle[:, self.a, :] + ratio[None] * state.angle[:, self.b, :]
        stress = k[None] * e + cubic[None] * e.pow(3)
        torque = torch.zeros_like(state.angle)
        batch = u.shape[0]
        idx_a = self.a[None, :, None].expand(batch, -1, self.channels)
        idx_b = self.b[None, :, None].expand(batch, -1, self.channels)
        torque = torque.scatter_add(1, idx_a, -stress)
        torque = torque.scatter_add(1, idx_b, -ratio[None] * stress)
        target = u[:, :, None] * self.motor_gain[None] + self.motor_bias[None]
        motors = self.motor_stiffness * (target - state.angle[:, :self.n, :])
        torque = torch.cat((torque[:, :self.n, :] + motors, torque[:, self.n:, :]), 1)
        accel = (torque - self.damping * state.velocity) / self.inertia
        return state.velocity, accel

    def step(self, state: MotorState, u: torch.Tensor, dt: Optional[float] = None):
        h = self.dt if dt is None else float(dt)
        if h <= 0:
            raise ValueError('dt must be positive')
        _, accel = self.derivatives(state, u)
        velocity = state.velocity + h * accel
        angle = state.angle + h * velocity  # semi-implicit/symplectic Euler
        return MotorState(angle, velocity)

    def evolve(self, u, steps: int = 50, state: Optional[MotorState] = None):
        if steps < 0:
            raise ValueError('steps must be nonnegative')
        state = self.make_state(u) if state is None else state
        for _ in range(steps):
            state = self.step(state, u)
        return state

    def forward(self, u, steps=50, state=None):
        state = self.evolve(u, steps=steps, state=state)
        return self.readout(state.angle[:, self.v0_idx, :]).squeeze(-1), state

    @torch.no_grad()
    def browser_export(self):
        k, cubic, ratio = self.coefficients()
        return {
            'schema': 'hdr-motor-v0.3', 'n': self.n, 'groups': self.groups,
            'channels': self.channels, 'dt': self.dt, 'damping': self.damping,
            'inertia': self.inertia, 'motorStiffness': self.motor_stiffness,
            'edges': [{'a': a, 'b': b, 'kind': kind,
                       'k': k[i].cpu().tolist(), 'cubic': cubic[i].cpu().tolist(),
                       'ratio': ratio[i].cpu().tolist()}
                      for i, (a, b, kind, _) in enumerate(self.edges)],
            'gain': self.motor_gain.cpu().tolist(), 'bias': self.motor_bias.cpu().tolist(),
            'readoutWeight': self.readout.weight[0].cpu().tolist(),
            'readoutBias': self.readout.bias[0].item(),
            'description': 'Tunable elastic gears; undriven V0; offline backprop learned parameters.'
        }
