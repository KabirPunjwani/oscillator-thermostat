"""Ensemble-averaged histograms compared against an ideal Gaussian (Maxwellian)."""

import matplotlib.pyplot as plt
import numpy as np

from .params import Params, column
from .simulate import evolve_batch, initial_batch, phase_grid


def ensemble_samples(
    p: Params,
    u0,
    variable="u",
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    burn_frac=0.2,
    store_step=50,
):
    """Run an n_theta x n_phi phase ensemble for one u0; return pooled post-burn-in samples."""
    th0s, ph0s = phase_grid(n_theta, n_phi)
    j = column(variable)

    store_indices = np.arange(0, p.n_steps, store_step)
    n_store = len(store_indices)
    stored = np.empty((n_store, th0s.size), dtype=float)

    ptr = 0
    for k, Zb in evolve_batch(p, initial_batch(u0, th0s, ph0s, xi0, Gam0, beta0)):
        if ptr < n_store and k == store_indices[ptr]:
            stored[ptr, :] = Zb[:, j]
            ptr += 1

    burn_store = int(np.floor(burn_frac * n_store))
    return stored[burn_store:, :].ravel()


def plot_ensemble_avg_with_ideal_gaussian(
    p: Params,
    u0_vals,
    variable="u",
    bins=41,
    xlim=(-4, 4),
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    burn_frac=0.2,
    store_step=50,
    gaussian_mode="standard",  # "standard" -> N(0,1), "data" -> N(mu, sigma) from the samples
    label_data="Ensemble-avg counts",
    label_gauss=None,
    xlabel=None,
    title=None,
):
    """Histogram a variable over every (u0, theta0, phi0) realization and overlay a Gaussian.

    Returns (bin_centers, average_counts_per_realization).
    """
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    th0s, ph0s = phase_grid(n_theta, n_phi)
    M = th0s.size
    j = column(variable)

    N = p.n_steps
    burn_idx = int(burn_frac * N)

    counts_sum = np.zeros(bins, dtype=float)
    n_realizations = 0
    s1 = s2 = 0.0
    n_samp = 0

    for u0 in u0_vals:
        samples = []
        for k, Zb in evolve_batch(p, initial_batch(u0, th0s, ph0s, xi0, Gam0, beta0)):
            if k >= burn_idx and k % store_step == 0:
                samples.append(Zb[:, j].copy())
        if not samples:
            continue

        S = np.vstack(samples)  # (n_samples_post_burn, M)
        s1 += np.sum(S)
        s2 += np.sum(S**2)
        n_samp += S.size

        # histograms are additive, so one call equals summing the M per-realization histograms
        counts_sum += np.histogram(S.ravel(), bins=edges)[0]
        n_realizations += M

    if n_realizations == 0:
        raise RuntimeError(
            f"No realizations collected. Check: burn_idx={burn_idx}, N={N}, store_step={store_step}."
        )

    avg_counts = counts_sum / n_realizations

    if gaussian_mode == "standard":
        mu, sigma = 0.0, 1.0
        label_gauss = label_gauss or "Ideal Gaussian (μ=0, σ=1)"
    elif gaussian_mode == "data":
        mu = s1 / n_samp
        sigma = np.sqrt(max((s2 / n_samp) - mu**2, 0.0)) or 1e-15
        label_gauss = label_gauss or f"Gaussian from data (μ={mu:.3g}, σ={sigma:.3g})"
    else:
        raise ValueError("gaussian_mode must be 'standard' or 'data'")

    # Gaussian overlay scaled to the histogram's counts
    x = np.linspace(xlim[0], xlim[1], 1000)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    gauss_counts = avg_counts.sum() * pdf * bin_width

    plt.figure(figsize=(8, 5))
    plt.plot(centers, avg_counts, "x", label=f"{label_data} ({n_realizations} realizations)")
    plt.plot(x, gauss_counts, "r", linewidth=2, label=label_gauss)
    plt.xlabel(xlabel or variable)
    plt.ylabel("Counts per bin")
    plt.title(title or f"Ensemble-averaged histogram for {variable} + Gaussian overlay, {p.label}")
    plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()

    return centers, avg_counts


def plot_data_with_ideal_maxwellian(
    data,
    bins=41,
    xlim=(-4, 4),
    label_data="Data",
    label_gauss="Ideal Maxwellian (μ=0, σ=1)",
    title="Histogram with ideal Maxwellian overlay",
):
    """Histogram raw samples (counts) and overlay an N(0,1) curve. No averaging or fitting."""
    data = np.asarray(data, dtype=float)

    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    counts, _ = np.histogram(data, bins=edges)

    x = np.linspace(xlim[0], xlim[1], 1000)
    pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    gauss_counts = counts.sum() * pdf * bin_width

    plt.figure(figsize=(8, 5))
    plt.plot(centers, counts, "x", label=label_data)
    plt.plot(x, gauss_counts, "r", linewidth=2, label=label_gauss)
    plt.xlabel("Value")
    plt.ylabel("Counts per bin")
    plt.title(title)
    plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()

    return centers, counts
