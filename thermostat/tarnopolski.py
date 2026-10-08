"""Tarnopolski plane: Abbe value vs turning-point fraction, to tell chaos from periodic/noise.

A time series is subsampled at several strides m. For each stride we compute
  - the Abbe value  A = <(x[i+1] - x[i])^2> / (2 var(x))
  - the turning-point fraction tau (share of interior points that are local extrema)
and trace the path (A, tau) as m grows. Periodic signals, white noise and chaotic
signals follow visibly different paths, so the coupled system's path is drawn
against three references: a sine, white noise, and a pure Van der Pol oscillator.

Both statistics are invariant to rescaling the series, which is why the pure-VdP
reference does not need the q (k) amplitude parameter.
"""

import matplotlib.pyplot as plt
import numpy as np

from .integrators import rk4_step
from .params import Params, column
from .simulate import DEFAULT_Z0, integrate

DEFAULT_M_LIST = [1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 80, 100, 120, 150, 170, 200]


def abbe_value(x):
    """Abbe value A = mean(diff(x)^2) / (2 var(x)); NaN if x is too short or constant."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    var = np.var(x, ddof=0)
    if var == 0:
        return np.nan
    return np.mean(np.diff(x) ** 2) / (2.0 * var)


def turning_point_fraction(x):
    """Fraction of interior points that are local maxima or minima."""
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    d1 = x[1:-1] - x[:-2]
    d2 = x[2:] - x[1:-1]
    return np.sum(d1 * d2 < 0) / (x.size - 2)


def tarnopolski_path(series, burn_frac=0.2, m_list=None, use_T_over_mu=False):
    """Trace (A, tau) over subsampling strides m_list after discarding burn-in.

    Returns (A_values, Y_values, m_list), where Y is tau, or 1.5*tau if use_T_over_mu.
    """
    s = np.asarray(series, dtype=float)
    s = s[int(burn_frac * len(s)):]
    m_list = DEFAULT_M_LIST if m_list is None else m_list

    A_vals, Y_vals = [], []
    for m in m_list:
        x = s[::m]
        tau_val = turning_point_fraction(x)
        A_vals.append(abbe_value(x))
        Y_vals.append(1.5 * tau_val if use_T_over_mu else tau_val)
    return np.array(A_vals), np.array(Y_vals), np.array(m_list)


def simulate_pure_vdp(p: Params, mu=1.0, z0=DEFAULT_Z0):
    """Pure Van der Pol reference in standard form: u' = mu (1 - xi^2) u - xi.

    Returns (tau, Z) with the same (N, 6) layout as the full system; the thermostat
    columns are zero.
    """
    N = p.n_steps
    tau = np.linspace(p.t0, p.t1, N)
    Z = np.zeros((N, 6), dtype=float)
    Z[0] = z0

    def fun(t, z):
        xi, u = z[0], z[1]
        return np.array([u, mu * (1.0 - xi**2) * u - xi, 0.0, 0.0, 0.0, 0.0])

    for k in range(N - 1):
        Z[k + 1] = rk4_step(fun, p.dt, tau[k], Z[k])
    return tau, Z


def plot_T_vs_A(
    p: Params,
    variable="u",
    burn_frac=0.2,
    m_list=None,
    use_T_over_mu=False,
    mu=1.0,
    seed=42,
):
    """Plot the Tarnopolski plane for the full MKT+VdP system against three references.

    p        parameters of the full system (omega_tau0, alpha, q, f0, t0, t1, dt)
    variable which state variable to analyse ("xi", "u", "Gam", ...)
    mu       Van der Pol strength of the pure-VdP reference curve
    seed     seed for the white-noise reference

    Returns ((A, Y) for sine, white noise, pure VdP, full system).
    """
    j = column(variable)
    N = p.n_steps
    tau_arr = np.linspace(p.t0, p.t1, N)

    def path(series):
        A, Y, _ = tarnopolski_path(series, burn_frac, m_list, use_T_over_mu)
        return A, Y

    sine = path(np.sin(tau_arr))
    noise = path(np.random.RandomState(seed).randn(N))
    _, Z_vdp = simulate_pure_vdp(p, mu)
    vdp = path(Z_vdp[:, j])
    full = path(integrate(p).Z[:, j])

    plt.figure(figsize=(6, 5))
    plt.plot(*sine, color="green", label="sine")
    plt.plot(*noise, color="blue", label="white noise")
    plt.plot(*vdp, color="orange", label=f"pure VdP (mu = {mu:g})")
    plt.plot(*full, color="red", label=f"MKT+VdP (ωτ={p.omega_tau0:g}, α={p.alpha:g}, q={p.q:g})")
    plt.xlabel("Abbe value A")
    plt.ylabel("T/μT" if use_T_over_mu else "τ (turning-point fraction)")
    plt.title(f"Tarnopolski plane — {variable}, ωτ={p.omega_tau0:g}, α={p.alpha:g}, q={p.q:g}")
    plt.legend()
    plt.tight_layout()

    return sine, noise, vdp, full
