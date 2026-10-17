"""Three-factor mechanical equilibrium propagation (EP) without backpropagation.

In the free phase all shafts, including V0, relax passively. During *training
only*, a weak supervised torque proportional to prediction error is applied to
V0. Each link changes its OWN stiffness / gear ratio using the difference in
its locally measured strain energy between the two equilibrated phases.

This is a numerical emulation of a physical training procedure, not an actual
on-device learning circuit. EP requires sufficiently settled equilibria; a
truncated relaxation is a biased approximation. No autograd is used anywhere.
"""
from dataclasses import dataclass
import math

import torch

from .mechanics import GearNetwork, MotorState


@dataclass
class LocalUpdate:
    grad_log_k: torch.Tensor
    grad_ratio_logits: torch.Tensor
    grad_log_cubic: torch.Tensor
    free_output: torch.Tensor
    nudged_output: torch.Tensor
    free_residual: float
    nudged_residual: float


@torch.no_grad()
def relaxed_phase(model: GearNetwork, input_targets: torch.Tensor, *, steps: int = 300,
                  state: MotorState | None = None, beta: float = 0.0,
                  teacher: torch.Tensor | None = None) -> MotorState:
    """Symplectic inertial relaxation with teacher spring ONLY in nudged phase."""
    if steps < 1 or beta < 0 or beta > 2:
        raise ValueError('Invalid relaxation steps or teacher stiffness')
    if beta > 0 and (teacher is None or teacher.shape != (len(input_targets), model.channels)):
        raise ValueError('Teacher must have shape [batch, channels]')
    state = model.make_state(input_targets) if state is None else MotorState(state.angle.clone(), state.velocity.clone())
    for _ in range(steps):
        _, accel = model.derivatives(state, input_targets)
        if beta:
            # Teacher torque has units of torque; zero at equilibrium if output matches target.
            # V0 is otherwise passive and gets no input-target drive.
            correction = (beta / model.inertia) * (teacher - state.angle[:, model.v0_idx, :])
            accel[:, model.v0_idx, :] += correction
        velocity = state.velocity + model.dt * accel
        angle = state.angle + model.dt * velocity
        state = MotorState(angle, velocity)
    if not torch.isfinite(state.angle).all() or not torch.isfinite(state.velocity).all():
        raise FloatingPointError('Relaxation diverged')
    return state


@torch.no_grad()
def residual(model: GearNetwork, input_targets: torch.Tensor, state: MotorState,
             beta: float = 0.0, teacher: torch.Tensor | None = None) -> float:
    _, accel = model.derivatives(state, input_targets)
    if beta:
        accel[:, model.v0_idx, :] += (beta / model.inertia) * (teacher - state.angle[:, model.v0_idx, :])
    return float((accel.square().mean()).sqrt())


@torch.no_grad()
def local_phase_gradients(model: GearNetwork, free: MotorState, nudged: MotorState,
                          beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local EP gradients of E with respect to physical parameters.

    No global chain rule or unrolled dynamics. For each edge (a,b):
       e = theta_a + r theta_b
       dE/d(log k) = (k/2)e²
       dE/d(log q) = (q/4)e⁴
       dE/d(ratio_logits) = (k e + q e³)*theta_b * sign*span*(1-tanh(logit)²)
    Contrast: (dE_nudged - dE_free)/beta; averaged across examples.
    """
    if beta <= 0:
        raise ValueError('beta must be positive')
    k, q, r = model.coefficients()
    def derivatives(state):
        e = state.angle[:, model.a, :] + r[None] * state.angle[:, model.b, :]
        theta_b = state.angle[:, model.b, :]
        dk = k[None] * e.square() / 2
        dq = q[None] * e.pow(4) / 4
        dr = ((k[None] * e + q[None] * e.pow(3)) * theta_b
              * (model.orientation[:, None] * model.ratio_span
                 * (1 - torch.tanh(model.ratio_logits).square()))[None])
        return dk, dr, dq
    d0, d1 = derivatives(free), derivatives(nudged)
    return tuple((b - a).mean(0) / beta for a, b in zip(d0, d1))


@torch.no_grad()
def local_update(model: GearNetwork, targets: torch.Tensor, labels: torch.Tensor,
                 *, free_steps: int = 300, nudged_steps: int = 300,
                 beta: float = 0.12, lr: float = 0.06, target_amplitude: float = 0.6,
                 train_kinds: tuple[str, ...] = ('pair', 'neighbor', 'group', 'hub'),
                 fit: bool = True, train_ratios: bool = True, teacher_angles: torch.Tensor | None = None) -> LocalUpdate:
    """One minibatch: learn ONLY link-local parameters from two phase strains.

    labels ∈ {0,1}. A scalar desired hub displacement is supplied *only* during
    training. This uses a supervised teacher signal; it is not unsupervised or
    autonomous learning. Each link's update uses only local endpoint angles and
    its own gear constants; no error derivatives propagate over the graph.
    """
    if model.channels != 1:
        raise ValueError('The local EP demonstration currently requires one physical channel')
    if labels.ndim != 1 or labels.shape[0] != targets.shape[0]:
        raise ValueError('labels must be [batch]')
    if not (0 < beta <= 2 and 0 < lr <= 1 and 0 < target_amplitude <= 2):
        raise ValueError('Invalid hyperparameters')
    if not set(train_kinds).issubset({'pair', 'neighbor', 'group', 'hub'}):
        raise ValueError('Unsupported link class')
    teacher = ((2 * labels.float() - 1)[:, None] * target_amplitude if teacher_angles is None
               else teacher_angles.reshape(-1, 1))
    if not torch.isfinite(teacher).all() or teacher.shape != (len(targets), 1):
        raise ValueError("Invalid teacher target angles")
    free = relaxed_phase(model, targets, steps=free_steps)
    # Start the nudged phase from the free equilibrium (not zero state).
    nudged = relaxed_phase(model, targets, state=free, steps=nudged_steps, beta=beta, teacher=teacher)
    grads = local_phase_gradients(model, free, nudged, beta)
    if fit:
        mask = torch.tensor([kind in train_kinds for kind in model.edge_kinds],
                            device=targets.device)[:, None]
        # Clipped per-link updates: robust step-size / avoid out-of-range physical gears.
        model.log_k.add_(-lr * grads[0].clamp(-2,2)*mask)
        model.log_k.clamp_(-6,1.8)
        if train_ratios:
            model.ratio_logits.add_(-lr * grads[1].clamp(-2,2)*mask)
            model.ratio_logits.clamp_(-3,3)
        model.log_cubic.add_(-lr * grads[2].clamp(-1,1)*mask)
        model.log_cubic.clamp_(-9,-0.6)
    return LocalUpdate(grads[0], grads[1], grads[2], free.angle[:,model.v0_idx,0],
                       nudged.angle[:,model.v0_idx,0], residual(model,targets,free),
                       residual(model,targets,nudged,beta,teacher))


def scalar_hub_accuracy(output: torch.Tensor, labels: torch.Tensor, polarity: int = 1) -> float:
    """Sign-only physical sensor, no learned classifier. Polarity fixed in advance."""
    return float((((output * polarity) >= 0).float() == labels).float().mean())


@torch.no_grad()
def predict_local(model: GearNetwork, targets: torch.Tensor, steps: int = 300):
    state = relaxed_phase(model, targets, steps=steps)
    return state.angle[:, model.v0_idx, 0]
