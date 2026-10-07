"""The six coupled ODEs: Van der Pol / SHO oscillator + MKT chaotic thermostat.

State vector: z = [xi, u, Gam, beta, th, ph]
    xi   oscillator position
    u    oscillator velocity (xi')
    Gam  fluctuating friction supplied by the thermostat
    beta thermostat feedback variable
    th, ph  phase variables driven by Gam and beta
"""

import numpy as np

from .params import Params


def rhs(tau, z, p: Params):
    """Right-hand side for a single trajectory z of shape (6,)."""
    xi, u, Gam, beta, th, ph = z
    inv_w2 = 1 / p.omega_tau0**2

    xi_dot = u
    u_dot = -xi + p.f0 * (np.sin(th) + np.sin(ph)) - Gam * u + p.alpha * (1 - (p.q**2) * (xi**2)) * u
    Gam_dot = inv_w2 * (u**2 - 1.0) - beta * Gam
    beta_dot = Gam**2 - inv_w2
    th_dot = Gam
    ph_dot = beta

    return np.array([xi_dot, u_dot, Gam_dot, beta_dot, th_dot, ph_dot], dtype=float)


def rhs_batch(tau, Z, p: Params):
    """Vectorized right-hand side for a batch Z of shape (M, 6): M trajectories at once."""
    xi, u, Gam, beta, th, ph = (Z[:, i] for i in range(6))
    inv_w2 = 1 / p.omega_tau0**2

    xi_dot = u
    u_dot = -xi + p.f0 * (np.sin(th) + np.sin(ph)) - Gam * u + p.alpha * (1 - (p.q**2) * (xi**2)) * u
    Gam_dot = inv_w2 * (u**2 - 1.0) - beta * Gam
    beta_dot = Gam**2 - inv_w2
    th_dot = Gam
    ph_dot = beta

    return np.column_stack([xi_dot, u_dot, Gam_dot, beta_dot, th_dot, ph_dot])
