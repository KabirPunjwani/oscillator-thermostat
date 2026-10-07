"""Time-domain pulse width: estimate tau_L from a Lorentzian-shaped pulse in x(tau)."""

import matplotlib.pyplot as plt
import numpy as np

from .params import Params


def _fwhm(tt, y, i0):
    """Full width at half maximum of the pulse peaking at index i0, or None if it is cut off."""
    half = 0.5 * y[i0]

    il = i0
    while il > 0 and y[il] > half:
        il -= 1
    ir = i0
    while ir < len(y) - 1 and y[ir] > half:
        ir += 1

    if il == 0 or ir == len(y) - 1:
        return None

    # linear interpolation for the two half-maximum crossings
    t_left = tt[il] + (half - y[il]) * (tt[il + 1] - tt[il]) / (y[il + 1] - y[il])
    t_right = tt[ir - 1] + (half - y[ir - 1]) * (tt[ir] - tt[ir - 1]) / (y[ir] - y[ir - 1])
    return t_right - t_left


def lorentzian_width_from_timeseries(t, x, tmin=None, tmax=None, use_abs=True):
    """Estimate the Lorentzian width of the most prominent pulse in x(t).

    Returns (t_peak, fwhm, tau_L) with tau_L = fwhm/2 for A / (1 + ((t - t_peak)/tau_L)^2).
    """
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax

    tt = t[mask]
    y = np.abs(x[mask]) if use_abs else x[mask]

    i0 = np.argmax(y)
    fwhm = _fwhm(tt, y, i0)
    if fwhm is None:
        return tt[i0], np.nan, np.nan
    return tt[i0], fwhm, fwhm / 2.0


def plot_pulse_with_lorentzian_overlay(t, x, p: Params, t_center=None, window=20.0, use_abs=True):
    """Plot a pulse in x(t) with a signed Lorentzian of matching width overlaid. Returns tau_L."""
    if t_center is None:
        t_center = t[np.argmax(np.abs(x))] if use_abs else t[np.argmax(x)]

    mask = (t >= t_center - window / 2) & (t <= t_center + window / 2)
    tt, xx = t[mask], x[mask]

    y = np.abs(xx) if use_abs else xx
    i0 = np.argmax(y)
    t_peak = tt[i0]
    peak_val = xx[i0]  # signed peak value

    fwhm = _fwhm(tt, y, i0)
    if fwhm is None:
        print("Pulse not fully contained in window; increase window.")
        return np.nan
    tau_L = fwhm / 2.0

    lor = peak_val / (1.0 + ((tt - t_peak) / tau_L) ** 2)

    plt.figure(figsize=(7, 4))
    plt.plot(tt, xx, label="x(tau) segment")
    plt.plot(tt, lor, "r", linewidth=2, label=f"Lorentzian fit, tau_L={tau_L:.3g}")
    plt.xlabel("tau")
    plt.ylabel("x")
    plt.title(f"Time-domain pulse and signed Lorentzian overlay, {p.label}")
    plt.legend()
    plt.tight_layout()

    print("t_peak =", t_peak)
    print("peak_val =", peak_val)
    print("FWHM =", fwhm)
    print("tau_L(time) =", tau_L)
    return tau_L
