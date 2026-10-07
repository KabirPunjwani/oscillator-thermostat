"""Running time averages and running kurtosis, averaged over an ensemble of realizations."""

import matplotlib.pyplot as plt
import numpy as np

from .params import Params
from .simulate import evolve_batch, initial_batch, phase_grid


def _x2(Z):
    return Z[:, 0] * Z[:, 0]


def _v2(Z):
    return Z[:, 1] * Z[:, 1]


def _v4(Z):
    vv = Z[:, 1] * Z[:, 1]
    return vv * vv


def _gam(Z):
    return Z[:, 2]


def _running_averages(p: Params, Zb0, observables, burn_frac, store_step):
    """Running time average (1/(t - t_burn)) * integral of f(Z) dt, per realization.

    observables maps a name to a function f(Z) -> array of shape (M,).
    Returns (t_store, {name: array of shape (n_store, M)}).
    """
    N = p.n_steps
    burn_idx = int(burn_frac * N)
    t_burn = p.t0 + burn_idx * p.dt

    post_indices = np.arange(0, N - burn_idx, store_step)  # indices counted from the end of burn-in
    n_store = len(post_indices)
    t_store = t_burn + post_indices * p.dt
    M = Zb0.shape[0]

    integrals = {name: np.zeros(M) for name in observables}
    stored = {name: np.zeros((n_store, M)) for name in observables}

    post_k = 0
    ptr = 0
    for k, Zb in evolve_batch(p, Zb0):
        if k >= burn_idx:
            for name, f in observables.items():  # rectangle rule on [t, t+dt)
                integrals[name] += f(Zb) * p.dt
            if ptr < n_store and post_k == post_indices[ptr]:
                elapsed = (post_k + 1) * p.dt  # time since burn-in, including this step
                for name in observables:
                    stored[name][ptr, :] = integrals[name] / elapsed
                ptr += 1
            post_k += 1

    return t_store, stored


def ensemble_running_time_averages_x2_v2(
    p: Params,
    u0_vals,
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    th0_shift=0.0,
    ph0_shift=0.0,
    burn_frac=0.2,
    store_step=10,
    show_plot=True,
):
    """Ensemble mean of the running time averages of x^2, v^2 and Gamma.

    Returns (t_store, mean_x2bar, mean_v2bar, mean_Gambar).
    """
    th0s, ph0s = phase_grid(n_theta, n_phi, th0_shift, ph0_shift)
    M = th0s.size
    observables = {"x2": _x2, "v2": _v2, "Gam": _gam}

    sums = None
    R_total = 0
    for u0 in u0_vals:
        Zb0 = initial_batch(u0, th0s, ph0s, xi0, Gam0, beta0)
        t_store, stored = _running_averages(p, Zb0, observables, burn_frac, store_step)
        batch = {name: arr.mean(axis=1) * M for name, arr in stored.items()}
        sums = batch if sums is None else {name: sums[name] + batch[name] for name in batch}
        R_total += M

    mean_x2bar = sums["x2"] / R_total
    mean_v2bar = sums["v2"] / R_total
    mean_Gambar = sums["Gam"] / R_total

    if show_plot:
        plt.figure(figsize=(9, 4))
        plt.plot(t_store, mean_x2bar, label=r"$\langle \overline{x^2}(t)\rangle$")
        plt.plot(t_store, mean_v2bar, label=r"$\langle \overline{v^2}(t)\rangle$")
        plt.plot(t_store, mean_x2bar / mean_v2bar, label=r"$\langle \overline{ratio}(t)\rangle$")
        plt.plot(t_store, mean_Gambar, label=r"$\langle \overline{\Gamma}(t)\rangle$")
        plt.axhline(p.q, ls="--", color="gray", label=f"q = {p.q:g} (expected $\\langle\\Gamma\\rangle$)")
        plt.xlabel(r"$\tau$")
        plt.ylabel("running time average")
        plt.title(f"Ensemble mean of running time-averaged $x^2$ and $v^2$, {p.label}")
        plt.legend()
        plt.tight_layout()

    return t_store, mean_x2bar, mean_v2bar, mean_Gambar


def ensemble_running_kurtosis_v(
    p: Params,
    u0_vals,
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    th0_shift=0.0,
    ph0_shift=0.0,
    burn_frac=0.2,
    store_step=10,
    show_plot=True,
    eps=1e-15,  # guards against division by zero
):
    """Ensemble mean of the running kurtosis K = <v^4> / <v^2>^2 of the velocity.

    K is computed per realization from its own running averages, then averaged.
    A Gaussian gives K = 3. Returns (t_store, mean_K, mean_v2bar, mean_v4bar).
    """
    th0s, ph0s = phase_grid(n_theta, n_phi, th0_shift, ph0_shift)
    M = th0s.size
    observables = {"v2": _v2, "v4": _v4}

    sum_K = sum_v2 = sum_v4 = 0.0
    R_total = 0
    for u0 in u0_vals:
        Zb0 = initial_batch(u0, th0s, ph0s, xi0, Gam0, beta0)
        t_store, stored = _running_averages(p, Zb0, observables, burn_frac, store_step)
        K = stored["v4"] / (stored["v2"] ** 2 + eps)  # (n_store, M)
        sum_K = sum_K + K.mean(axis=1) * M
        sum_v2 = sum_v2 + stored["v2"].mean(axis=1) * M
        sum_v4 = sum_v4 + stored["v4"].mean(axis=1) * M
        R_total += M

    mean_K = sum_K / R_total
    mean_v2bar = sum_v2 / R_total
    mean_v4bar = sum_v4 / R_total

    if show_plot:
        plt.figure(figsize=(9, 4))
        plt.plot(t_store, mean_K, label=r"$K(\tau)=\langle \overline{v^4}\rangle / \langle \overline{v^2}\rangle^2$")
        plt.xlabel(r"$\tau$")
        plt.ylabel("kurtosis")
        plt.title(f"Ensemble running kurtosis of velocity, {p.label}")
        plt.legend()
        plt.tight_layout()

    return t_store, mean_K, mean_v2bar, mean_v4bar
