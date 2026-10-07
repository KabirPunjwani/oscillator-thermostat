"""Running the system: one trajectory, or an ensemble of trajectories in parallel."""

from dataclasses import dataclass

import numpy as np

from .integrators import rk4_step
from .params import Params
from .system import rhs, rhs_batch

# Default initial state: xi=0, u=1, Gam=0, beta=0, th=0, ph=0
DEFAULT_Z0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


@dataclass
class Trajectory:
    """A single integrated trajectory: time axis plus the (N, 6) state history."""

    tau: np.ndarray
    Z: np.ndarray

    xi = property(lambda self: self.Z[:, 0])
    u = property(lambda self: self.Z[:, 1])
    Gam = property(lambda self: self.Z[:, 2])
    beta = property(lambda self: self.Z[:, 3])
    th = property(lambda self: self.Z[:, 4])
    ph = property(lambda self: self.Z[:, 5])


def integrate(p: Params, z0=DEFAULT_Z0) -> Trajectory:
    """Integrate one trajectory from z0 with RK4."""
    N = p.n_steps
    tau = np.linspace(p.t0, p.t1, N)
    Z = np.zeros((N, 6), dtype=float)
    Z[0] = z0

    def fun(t, z):
        return rhs(t, z, p)

    for k in range(N - 1):
        Z[k + 1] = rk4_step(fun, p.dt, tau[k], Z[k])
    return Trajectory(tau, Z)


def phase_grid(n_theta, n_phi, th0_shift=0.0, ph0_shift=0.0):
    """Uniform grid of initial phases (theta0, phi0) on [0, 2pi); returns flat arrays."""
    thetas = (np.linspace(0, 2 * np.pi, n_theta, endpoint=False) + th0_shift) % (2 * np.pi)
    phis = (np.linspace(0, 2 * np.pi, n_phi, endpoint=False) + ph0_shift) % (2 * np.pi)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    return TH.ravel(), PH.ravel()


def initial_batch(u0, th0s, ph0s, xi0=0.0, Gam0=0.0, beta0=0.0):
    """Initial state for M parallel trajectories sharing u0 but with different phases."""
    Zb = np.zeros((th0s.size, 6), dtype=float)
    Zb[:, 0] = xi0
    Zb[:, 1] = u0
    Zb[:, 2] = Gam0
    Zb[:, 3] = beta0
    Zb[:, 4] = th0s
    Zb[:, 5] = ph0s
    return Zb


def evolve_batch(p: Params, Zb):
    """Step a batch forward in time, yielding (k, Zb) at every step k = 0 .. N-1.

    The state is yielded *before* it is advanced, so step k always sees the state
    at time t0 + k*dt. Callers should copy Zb if they keep it.
    """
    N = p.n_steps
    t = p.t0

    def fun(tt, Z):
        return rhs_batch(tt, Z, p)

    for k in range(N):
        yield k, Zb
        if k < N - 1:
            Zb = rk4_step(fun, p.dt, t, Zb)
            t += p.dt
