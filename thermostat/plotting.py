"""Single-trajectory plots: time series and phase portraits."""

import matplotlib.pyplot as plt
import numpy as np

from .params import Params
from .simulate import Trajectory


def _tail(traj: Trajectory, span):
    """Default x-window: the last `span` units of the run."""
    return max(traj.tau[0], traj.tau[-1] - span), traj.tau[-1]


def _series_plot(traj, p, y, ylabel, name, xlim, default_span, label=None):
    plt.figure(figsize=(9, 4))
    plt.plot(traj.tau, y, label=label or f"{name}(tau)")
    plt.xlabel("tau")
    plt.ylabel(ylabel)
    plt.xlim(*(xlim or _tail(traj, default_span)))
    plt.title(f"{name} vs tau, {p.label}")
    plt.legend()
    plt.tight_layout()


def xi_plot(traj: Trajectory, p: Params, xlim=None):
    _series_plot(traj, p, traj.xi, "xi", "xi", xlim, 100)


def u_plot(traj: Trajectory, p: Params, xlim=None):
    _series_plot(traj, p, traj.u, "u", "u", xlim, 17)


def gam_plot(traj: Trajectory, p: Params, xlim=None):
    _series_plot(traj, p, traj.Gam, "Gam", "Gam", xlim, 100)


def q_minus_gamma_plot(traj: Trajectory, p: Params, xlim=None):
    _series_plot(traj, p, p.q - traj.Gam, "q-Gam", "q-Gam", xlim, 100)


def gam_u_series_plot(traj: Trajectory, p: Params, xlim=None):
    plt.figure(figsize=(9, 4))
    plt.plot(traj.tau, traj.Gam, label=f"Gam(tau), q={p.q:g}")
    plt.plot(traj.tau, traj.u, label="u(tau)", alpha=0.8)
    plt.xlabel("tau")
    plt.ylabel("Gam/u")
    plt.xlim(*(xlim or _tail(traj, 100)))
    plt.legend()
    plt.tight_layout()


def _phase_plot(traj, p, x, y, xlabel, t_start, t_end, equal_axes):
    t_start = traj.tau[0] if t_start is None else t_start
    t_end = traj.tau[-1] if t_end is None else t_end
    i0, i1 = np.searchsorted(traj.tau, [t_start, t_end])
    plt.figure(figsize=(5, 5))
    plt.plot(x[i0:i1], y[i0:i1])
    plt.xlabel(xlabel)
    plt.ylabel("xi_dot or u'")
    plt.title(f"Phase portrait, {p.label}")
    if equal_axes:
        plt.axis("equal")
    plt.tight_layout()


def phase_portrait(traj: Trajectory, p: Params, t_start=None, t_end=None):
    """xi vs u over [t_start, t_end] (defaults to the whole run)."""
    _phase_plot(traj, p, traj.xi, traj.u, "xi", t_start, t_end, equal_axes=True)


def phase_portrait_gam_u(traj: Trajectory, p: Params, t_start=None, t_end=None):
    """Gam vs u over [t_start, t_end] (defaults to the whole run)."""
    _phase_plot(traj, p, traj.Gam, traj.u, "Gam", t_start, t_end, equal_axes=False)
