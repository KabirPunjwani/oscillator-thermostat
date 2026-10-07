"""Fourier analysis: Hann-windowed spectra, exponential/power-law fits, ensemble spectra."""

import matplotlib.pyplot as plt
import numpy as np

from .params import Params, column
from .simulate import Trajectory, evolve_batch, initial_batch, phase_grid


def one_sided_spectrum(x, dt, use_hann=True, detrend=True):
    """One-sided magnitude spectrum |X(omega)| for omega >= 0.

    x may be 1-D (one signal) or 2-D (time on axis 0, one column per realization).
    The DC offset is removed, a Hann window is applied, and the amplitude is
    corrected by the window's mean (coherent gain). dt/w_mean makes the discrete
    sum approximate the continuous Fourier transform.
    """
    x = np.asarray(x, dtype=float)
    if detrend:
        x = x - x.mean(axis=0, keepdims=True)

    n = x.shape[0]
    if use_hann:
        w = np.hanning(n)
        w_mean = np.mean(w)
    else:
        w = np.ones(n)
        w_mean = 1.0

    xw = x * (w if x.ndim == 1 else w[:, None])
    X = (dt / w_mean) * np.fft.fft(xw, axis=0)

    omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)  # rad per unit tau
    X_s = np.fft.fftshift(X, axes=0)
    omega_s = np.fft.fftshift(omega)

    pos = omega_s >= 0
    return omega_s[pos], np.abs(X_s)[pos]


def _require_points(mask, w1, w2):
    if mask.sum() < 2:
        raise ValueError(
            f"Fewer than 2 frequency points in the fit band [{w1}, {w2}]. "
            "The run is too short to resolve this band: increase t1 or adjust the band."
        )


def fit_exponential(omega, mag, w1, w2, eps=1e-300):
    """Fit ln|X| = a + b*omega on [w1, w2]. Returns (a, b); the timescale is tau_L = -b."""
    mask = (omega >= w1) & (omega <= w2) & (mag > 0)
    _require_points(mask, w1, w2)
    b, a = np.polyfit(omega[mask], np.log(mag[mask] + eps), 1)
    return a, b


def fit_power_law(omega, mag, w1, w2, eps=1e-300):
    """Fit |X| = C * omega**m on [w1, w2]. Returns (m, C)."""
    mask = (omega >= w1) & (omega <= w2) & (omega > 0) & (mag > 0)
    _require_points(mask, w1, w2)
    m, c = np.polyfit(np.log(omega[mask] + eps), np.log(mag[mask] + eps), 1)
    return m, np.exp(c)


def _plot_spectrum(omega, mag, test_mag, log, xlim, title):
    plt.figure(figsize=(9, 4))
    draw = plt.semilogy if log else plt.plot
    draw(omega, mag, label="VdP (one-sided)", linewidth=2)
    if test_mag is not None:
        draw(omega, test_mag, "--", label="sin(t) (one-sided)", linewidth=2)
    if xlim:
        plt.xlim(*xlim)
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|\tilde f(\omega)|$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def fourier_transform(
    traj: Trajectory,
    p: Params,
    start=None,
    end=None,
    diagnostic=False,
    log=True,
    fit_band=(0.1, 25.0),
    xlim=None,
):
    """Spectrum of u(tau) for one trajectory, plus an exponential fit of the tail.

    diagnostic=True overlays the spectrum of sin(t) so you can check the peak
    position and amplitude normalization of the pipeline.
    Returns (omega, magnitude, tau_L).
    """
    start = 0 if start is None else start
    end = len(traj.tau) if end is None else end
    t = traj.tau[start:end]
    dt = t[1] - t[0]

    omega, mag = one_sided_spectrum(traj.u[start:end], dt)
    print(f"The first peak is at {omega[np.argmax(mag)]} rad/s")

    test_mag = None
    if diagnostic:
        _, test_mag = one_sided_spectrum(np.sin(t), dt, detrend=False)
        print(f"sin(t) peak ω ≈ {omega[np.argmax(test_mag)]:.6f} rad/s (expect 1)")

    kind = "VdP vs sin(t)" if diagnostic else p.label
    _plot_spectrum(omega, mag, test_mag, log, xlim, f"One-sided magnitude spectrum (ω ≥ 0), Hann window, {kind}")

    w1, w2 = fit_band
    a, b = fit_exponential(omega, mag, w1, w2)
    tau_L = -b
    print("Fit results:")
    print(f"  slope b = {b:.6e}  (so tau_L = {-b:.6e})")
    print(f"  intercept a = {a:.6e}", end="\n\n")

    omega_line = np.linspace(w1, w2, 400)
    plt.figure(figsize=(9, 4))
    plt.semilogy(omega, mag, label=r"$|X(\omega)|$")
    plt.semilogy(omega_line, np.exp(a + b * omega_line), linewidth=3, label=f"fit: |X| = exp(a + b ω)\nτ_L ≈ {tau_L:.3f}")
    plt.xlim(w1, w2)
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel(r"$|\tilde X(\omega)|$")
    plt.title(f"Log-linear (semilog-y) spectrum with exponential-fit region, {p.label}")
    plt.legend()
    plt.tight_layout()

    return omega, mag, tau_L


def ensemble_spectrum(
    p: Params,
    u0_vals,
    variable="u",
    n_theta=1,
    n_phi=1,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    th0_shift=0.0,
    ph0_shift=0.0,
    burn_frac=0.2,
    use_hann=True,
):
    """Magnitude spectrum averaged over every (u0, theta0, phi0) realization.

    Returns (omega, mean_magnitude).
    """
    th0s, ph0s = phase_grid(n_theta, n_phi, th0_shift, ph0_shift)
    M = th0s.size
    j = column(variable)

    N = p.n_steps
    burn_idx = int(burn_frac * N)
    n_fft = N - burn_idx

    mag_sum = None
    n_total = 0
    omega = None

    for u0 in u0_vals:
        X = np.empty((n_fft, M), dtype=float)  # post-burn-in history, one column per realization
        ptr = 0
        for k, Zb in evolve_batch(p, initial_batch(u0, th0s, ph0s, xi0, Gam0, beta0)):
            if k >= burn_idx:
                X[ptr, :] = Zb[:, j]
                ptr += 1

        omega, mag = one_sided_spectrum(X, p.dt, use_hann=use_hann)  # (n_freq, M)
        batch_mean = mag.mean(axis=1)
        mag_sum = batch_mean * M if mag_sum is None else mag_sum + batch_mean * M
        n_total += M

    return omega, mag_sum / n_total


def ensemble_fft(
    p: Params,
    u0_vals,
    variable="u",
    mode="semilog",
    n_theta=1,
    n_phi=1,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    th0_shift=0.0,
    ph0_shift=0.0,
    burn_frac=0.2,
    use_hann=True,
    show_plot=True,
    w1=0.1,
    w2=20.0,
    pl_band=(1e-2, 1e-1),
    exp_band=(1e-1, 1.0),
    eps=1e-300,
):
    """Ensemble-averaged spectrum with one of three analysis modes.

    mode="semilog": semilog-y plot + exponential fit ln|X| = a + b*omega on [w1, w2];
                    returns tau_L = -b.
    mode="loglog":  log-log plot with two fits: a power law on pl_band and an
                    exponential envelope on exp_band; returns tau_L from the exponential.
    mode="linear":  plain linear-scale plot on [w1, w2]; no fit (tau_L is None).

    Returns (omega, mean_magnitude, tau_L).
    """
    if mode not in ("semilog", "loglog", "linear"):
        raise ValueError("mode must be 'semilog', 'loglog' or 'linear'")

    omega, mag = ensemble_spectrum(
        p, u0_vals, variable, n_theta, n_phi, xi0, Gam0, beta0, th0_shift, ph0_shift, burn_frac, use_hann
    )
    print(f"[ensemble avg] First peak at ω = {omega[np.argmax(mag)]} rad/s")

    base = f"for {variable}, {p.label}"
    ylabel = r"$|\tilde X(\omega)|$"
    curve = r"$\langle |X(\omega)| \rangle$"

    if mode == "linear":
        if show_plot:
            plt.figure(figsize=(9, 4))
            plt.plot(omega, mag, label=curve)
            plt.xlim(w1, w2)
            plt.xlabel(r"$\omega$ (rad/s)")
            plt.ylabel(ylabel)
            plt.title(f"Ensemble-averaged spectrum (linear scale) {base}")
            plt.legend()
            plt.tight_layout()
        return omega, mag, None

    if mode == "loglog":
        keep = (omega > 0) & (mag > 0)
        if show_plot:
            plt.figure(figsize=(9, 4))
            plt.loglog(omega[keep], mag[keep], label=curve)
            plt.xlim(1, 50)
            plt.xlabel(r"$\omega$ (rad/s)")
            plt.ylabel(ylabel)
            plt.title(f"Ensemble-averaged spectrum (full range, log-log) {base}")
            plt.legend()
            plt.tight_layout()

        m, C = fit_power_law(omega, mag, *pl_band, eps=eps)
        print("Power-law fit (log-log):")
        print(f"  band [{pl_band[0]},{pl_band[1]}]")
        print(f"  m = {m:.6e}   so |X| ~ C ω^m")
        print(f"  C = {C:.6e}", end="\n\n")

        a, b = fit_exponential(omega, mag, *exp_band, eps=eps)
        tau_L = -b
        print("Exponential-envelope fit (on log-linear in ω):")
        print(f"  band [{exp_band[0]},{exp_band[1]}]")
        print(f"  slope b = {b:.6e}  -> tau_L = {-b:.6e}")
        print(f"  intercept a = {a:.6e}", end="\n\n")

        if show_plot:
            w_pl = np.linspace(*pl_band, 400)
            w_exp = np.linspace(*exp_band, 400)
            plt.figure(figsize=(9, 4))
            plt.loglog(omega[keep], mag[keep], label=curve)
            plt.loglog(w_pl, C * w_pl**m, linewidth=3, label=f"power-law fit [{pl_band[0]},{pl_band[1]}]: m≈{m:.3g}")
            plt.loglog(
                w_exp, np.exp(a + b * w_exp), linewidth=3, label=f"exp envelope fit [{exp_band[0]},{exp_band[1]}]: τ_L≈{tau_L:.3g}"
            )
            plt.legend()
            plt.tight_layout()
        return omega, mag, tau_L

    # mode == "semilog"
    if show_plot:
        plt.figure(figsize=(9, 4))
        plt.semilogy(omega, mag, label=curve)
        plt.xlabel(r"$\omega$ (rad/s)")
        plt.ylabel(ylabel)
        plt.title(f"Ensemble-averaged spectrum (full range, semilog-y) {base}")
        plt.legend()
        plt.tight_layout()

    a, b = fit_exponential(omega, mag, w1, w2, eps=eps)
    tau_L = -b
    print("Fit results (ensemble avg):")
    print(f"  slope b = {b:.6e}  -> tau_L = {-b:.6e}")
    print(f"  intercept a = {a:.6e}", end="\n\n")

    if show_plot:
        omega_line = np.linspace(w1, w2, 400)
        plt.figure(figsize=(9, 4))
        plt.semilogy(omega, mag, label=curve)
        plt.semilogy(omega_line, np.exp(a + b * omega_line), linewidth=3, label=f"exp fit: τ_L≈{tau_L:.3g}")
        plt.xlim(w1, w2)
        plt.xlabel(r"$\omega$ (rad/s)")
        plt.ylabel(ylabel)
        plt.title(f"Ensemble-averaged spectrum with exponential fit (semilog-y) {base}")
        plt.legend()
        plt.tight_layout()

    return omega, mag, tau_L
