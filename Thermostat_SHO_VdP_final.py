#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:52:10 2026

@author: kabirpunjwani
"""

import numpy as np
import matplotlib.pyplot as plt


# ========= PARAMETERS =========
omega_tau0 = 10
f0 = 0.2 #0.35/omega_tau0
# Integration settings (tau-range)
alpha = 7
q = 0.5
t0, t1 = 0, 1200
dt   = 0.01

# ========= INITIAL CONDITIONS =========
z0 = np.array([
    0.0,   # xi(0)
    1.0,   # u(0) = xi'(0)
    0.0,   # Gamma(0)
    0.0,   # beta(0)
    0.0,   # theta(0)
    0.0    # phi(0)
], dtype=float)

# z0_c = np.array([
#     0.0,   # xi(0)
#     0.0,   # u(0) = xi'(0)
#     0.0,   # Gamma(0)
#     0.0,   # beta(0)
#     0.0,   # theta(0)
#     0.0    # phi(0)
# ], dtype=float)

# ========= RK4 SINGLE STEP =========
def rk4singlestep(fun, dt, t0, z0):
    f1 = fun(t0, z0)
    f2 = fun(t0 + dt/2, z0 + (dt/2)*f1)
    f3 = fun(t0 + dt/2, z0 + (dt/2)*f2)
    f4 = fun(t0 + dt,   z0 + dt*f3)
    return z0 + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)

# ========= SHO + MKT (6 ODEs) =========
def SHO_MKT_system(tau, z):
    xi, u, Gam, beta, th, ph = z

    xi_dot = u

    u_dot  = -xi + f0*(np.sin(th) + np.sin(ph)) - Gam*u + alpha*(1-(q**2)*(xi**2))*u

    Gam_dot  = (1/(omega_tau0)**2)*(u**2 - 1.0) - beta*Gam
    beta_dot = Gam**2 - (1/(omega_tau0)**2)

    th_dot = Gam
    ph_dot = beta

    return np.array([xi_dot, u_dot, Gam_dot, beta_dot, th_dot, ph_dot], dtype=float)


# ========= TIME MARCH WITH RK4 =========
N = int(np.ceil((t1 - t0)/dt)) + 1
tau = np.linspace(t0, t1, N)

Z = np.zeros((N, 6), dtype=float)
Z[0] = z0

for k in range(N-1):
    Z[k+1] = rk4singlestep(SHO_MKT_system, dt, tau[k], Z[k])

# ========= UNPACK =========
xi, u, Gam, beta, th, ph = Z.T

step = 10
indices = np.arange(0, len(tau), step)
xi_store = xi[indices]
u_store  = u[indices]
tau_store  = tau[indices]
Gam_store  = Gam[indices]
beta_store = beta[indices]


def xi_plot():
    plt.figure(figsize=(9, 4))
    # plt.plot(t[::k], x[::k], label="x(t)")
    plt.plot(tau, xi, label="xi(tau)")
    plt.xlabel("tau"); plt.ylabel("xi")
    plt.xlim(3700,4100)
    plt.title(f"xi vs tau,omegatau = {omega_tau0} and q {q}")
    plt.legend()
    plt.tight_layout()
    
def u_plot():
    plt.figure(figsize=(9, 4))
    plt.plot(tau, u, label="u(tau)")
    plt.xlabel("tau"); plt.ylabel("u")
    # plt.ylim(-0.06,0.06)
    plt.xlim(1100,1117)
    plt.title(f"u vs tau,omegatau = {omega_tau0} and q {q}")
    plt.legend()
    plt.tight_layout()
    
def q_minus_gamma_plot():
    plt.figure(figsize=(9, 4))
    # plt.plot(t[::k], x[::k], label="x(t)")
    plt.plot(tau, q-Gam, label="q-Gam(tau)")
    plt.xlabel("tau"); plt.ylabel("q-Gam")
    plt.xlim(1100,1200)
    plt.title(f"q-gam vs tau,omegatau = {omega_tau0} and q {q}")
    plt.legend()
    plt.tight_layout()
    
    
    
def Gam_u_series_plot():
    plt.figure(figsize=(9,4))
    plt.plot(tau, Gam,  label=f"Gam(tau), q={q}")
    plt.plot(tau, u, label="u(tau), q=0", alpha=0.8)
    plt.xlabel("tau"); plt.ylabel("Gam/u")
    plt.xlim(1100,1200)
    plt.legend(); plt.tight_layout(); plt.show() 
    
    
def gam_plot():
    plt.figure(figsize=(9, 4))
    plt.plot(tau, Gam, label="Gam(tau)")
    plt.xlabel("tau"); plt.ylabel("Gam")
    plt.xlim(1100,1200) 
    plt.title(f"Gam vs tau,omegatau = {omega_tau0} and q {q}")
    plt.legend()
    plt.tight_layout()



def phase_portrait(t1,t2):
    start_idx1 = np.searchsorted(tau, t1)
    start_idx2 = np.searchsorted(tau, t2)
    plt.figure(figsize=(5, 5))
    # plt.plot(x[burn::k], y[burn::k])
    plt.plot(xi[start_idx1:start_idx2], u[start_idx1:start_idx2]) 
    plt.xlabel("xi"); plt.ylabel("xi_dot or u'")
    plt.title(f"Phase portrait,omegatau = {omega_tau0} and q {q}")
    plt.axis("equal")
    # plt.xlim(-2.5, 2.5)     # <- critical
    # plt.ylim(-150, 150)     # <- critical
    plt.tight_layout()
    
    
def phase_portrait_gam_u(t1,t2):
    start_idx1 = np.searchsorted(tau, t1)
    start_idx2 = np.searchsorted(tau, t2)
    plt.figure(figsize=(5, 5))
    plt.plot(Gam[start_idx1:start_idx2], u[start_idx1:start_idx2]) 
    # plt.xlim(-1,1)
    # plt.ylim(-10,10)
    plt.xlabel("Gam"); plt.ylabel("xi_dot or u'")
    plt.title(f"Phase portrait,omegatau = {omega_tau0} and q {q}")
    plt.tight_layout()
    

def SHO_MKT_system_batch(tau, Z):
    """
    Vectorized RHS.
    Z shape: (M, 6) where columns are [xi, u, Gam, beta, th, ph]
    Returns dZ/dtau with same shape.
    """
    xi   = Z[:, 0]
    u    = Z[:, 1]
    Gam  = Z[:, 2]
    beta = Z[:, 3]
    th   = Z[:, 4]
    ph   = Z[:, 5]

    xi_dot = u
    u_dot  = -xi + f0*(np.sin(th) + np.sin(ph)) - Gam*u + alpha*(1-(q**2)*(xi**2))*u

    Gam_dot  = (1/(omega_tau0**2))*(u**2 - 1.0) - beta*Gam
    beta_dot = Gam**2 - (1/(omega_tau0**2))

    th_dot = Gam
    ph_dot = beta

    dZ = np.column_stack([xi_dot, u_dot, Gam_dot, beta_dot, th_dot, ph_dot])
    return dZ


def rk4_batch_step(fun, dt, t, Z):
    k1 = fun(t, Z)
    k2 = fun(t + dt/2, Z + (dt/2)*k1)
    k3 = fun(t + dt/2, Z + (dt/2)*k2)
    k4 = fun(t + dt,   Z + dt*k3)
    return Z + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def ensemble_u_samples_for_one_u0(
    u0,
    variable = "u",
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    burn_frac=0.2,
    store_step=50
):
    """
    Runs 20x20 trajectories in parallel for a fixed initial velocity u0.
    Stores u every store_step, discards burn-in, returns a flat array of u samples
    pooled across all realizations.
    """
    # 20 theta0, 20 phi0 (uniform in [0, 2pi))
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phis   = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    th0s = TH.ravel()
    ph0s = PH.ravel()
    M = th0s.size  # 400
    
    col_map = {"xi":0, "u":1, "Gam":2, "beta":3, "th":4, "ph":5}
    j = col_map[variable]

    # initial state batch: (M, 6)
    Zb = np.zeros((M, 6), dtype=float)
    Zb[:, 0] = xi0
    Zb[:, 1] = u0
    Zb[:, 2] = Gam0
    Zb[:, 3] = beta0
    Zb[:, 4] = th0s
    Zb[:, 5] = ph0s

    # time setup
    N = int(np.ceil((t1 - t0)/dt)) + 1
    burn_idx = int(burn_frac * N)

    # indices where we store (every store_step)
    store_indices = np.arange(0, N, store_step)
    n_store = len(store_indices)

    # store u(t) for all trajectories at store times
    U_store = np.empty((n_store, M), dtype=float)

    t = t0
    store_ptr = 0

    for k in range(N):
        if store_ptr < n_store and k == store_indices[store_ptr]:
            U_store[store_ptr, :] = Zb[:, j]
            store_ptr += 1

        if k < N - 1:
            Zb = rk4_batch_step(SHO_MKT_system_batch, dt, t, Zb)
            t += dt

    # burn-in in "stored-index space"
    burn_store = int(np.floor(burn_frac * n_store))
    U_post = U_store[burn_store:, :]

    # pool across all realizations (each has same number of stored samples)
    u_samples = U_post.ravel()
    return u_samples




def plot_ensemble_avg_with_ideal_gaussian(
    u0_vals,
    variable="u",                  # "xi", "u", "Gam", "beta", "th", "ph"
    bins=41,
    xlim=(-4, 4),
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    burn_frac=0.2,
    store_step=50,
    gaussian_mode="standard",      # "standard" -> N(0,1), "data" -> N(mu_data, sigma_data)
    label_data="Ensemble-avg counts",
    label_gauss=None,
    xlabel=None,
    title=None,
):
    # ---- histogram bins ----
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # ---- realization grid ----
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    phis   = np.linspace(0, 2*np.pi, n_phi,   endpoint=False)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    th0s = TH.ravel()
    ph0s = PH.ravel()
    M = th0s.size

    col_map = {"xi":0, "u":1, "Gam":2, "beta":3, "th":4, "ph":5}
    if variable not in col_map:
        raise ValueError(f"variable must be one of {list(col_map.keys())}")
    j = col_map[variable]

    # ---- time setup ----
    N = int(np.ceil((t1 - t0)/dt)) + 1
    burn_idx = int(burn_frac * N)

    counts_sum = np.zeros(bins, dtype=float)
    n_realizations_total = 0

    # for gaussian_mode="data"
    if gaussian_mode == "data":
        s1 = 0.0
        s2 = 0.0
        n_samp = 0

    for u0 in u0_vals:
        # batch ICs
        Zb = np.zeros((M, 6), dtype=float)
        Zb[:, 0] = xi0
        Zb[:, 1] = u0
        Zb[:, 2] = Gam0
        Zb[:, 3] = beta0
        Zb[:, 4] = th0s
        Zb[:, 5] = ph0s

        samples_list = []
        t = t0

        for k in range(N):
            # STORE rule: after burn-in, every store_step
            if k >= burn_idx and (k % store_step == 0):
                samples_list.append(Zb[:, j].copy())

            if k < N - 1:
                Zb = rk4_batch_step(SHO_MKT_system_batch, dt, t, Zb)
                t += dt

        if len(samples_list) == 0:
            continue

        S = np.vstack(samples_list)  # (n_store_post, M)

        if gaussian_mode == "data":
            s1 += np.sum(S)
            s2 += np.sum(S**2)
            n_samp += S.size

        # histogram each realization separately
        for m in range(M):
            c_m, _ = np.histogram(S[:, m], bins=edges, density=False)
            counts_sum += c_m
            n_realizations_total += 1

    if n_realizations_total == 0:
        raise RuntimeError(
            f"No realizations collected. Check: burn_idx={burn_idx}, N={N}, store_step={store_step}."
        )

    avg_counts = counts_sum / n_realizations_total

    # ---- Gaussian parameters ----
    if gaussian_mode == "standard":
        mu, sigma = 0.0, 1.0
        if label_gauss is None:
            label_gauss = "Ideal Gaussian (μ=0, σ=1)"
    elif gaussian_mode == "data":
        mu = s1 / n_samp
        var = (s2 / n_samp) - mu**2
        sigma = np.sqrt(max(var, 0.0))
        if sigma == 0:
            sigma = 1e-15
        if label_gauss is None:
            label_gauss = f"Gaussian from data (μ={mu:.3g}, σ={sigma:.3g})"
    else:
        raise ValueError("gaussian_mode must be 'standard' or 'data'")

    # ---- overlay (scaled to avg histogram counts) ----
    x = np.linspace(xlim[0], xlim[1], 1000)
    pdf = (1.0 / (sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    # pdf = (1.0 / (sigma*np.sqrt(2*np.pi))) * np.exp(-0.03125*((x-mu)/sigma)**2) #correction for Gamma

    total_counts_avg = avg_counts.sum()
    gauss_counts = total_counts_avg * pdf * bin_width

    if xlabel is None:
        xlabel = variable
    if title is None:
        title = f"Ensemble-averaged histogram for {variable} + Gaussian overlay, omegatau = {omega_tau0} and q {q}"

    plt.figure(figsize=(8, 5))
    plt.plot(centers, avg_counts, 'x',
              label=f"{label_data} ({n_realizations_total} realizations)")
    plt.plot(x, gauss_counts, 'r', linewidth=2, label=label_gauss)

    plt.xlabel(xlabel)
    plt.ylabel("Counts per bin")
    plt.title(title)
    plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return centers, avg_counts

#maxwellian cruve with only a single intial condition
def plot_data_with_ideal_maxwellian(
    data,
    bins=41,
    xlim=(-4, 4),
    label_data="Data",
    label_gauss="Ideal Maxwellian (μ=0, σ=1)"
):
    """
    Plot histogram of data (counts) and overlay ideal Maxwellian (Gaussian)
    with mean=0, sigma=1. No scaling of data, no averaging, no fitting.
    """

    data = np.asarray(data, dtype=float)

    # --- histogram (counts, NOT density) ---
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    counts, _ = np.histogram(data, bins=edges, density=False)

    # --- ideal Maxwellian (Gaussian) ---
    x = np .linspace(xlim[0], xlim[1], 1000)
    pdf = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5* x**2)
    # pdf = 4*(1.0 / np.sqrt(2*np.pi)) * np.exp(-8 * x**2) #this is for gamma

    # scale Gaussian to histogram counts
    total_counts = counts.sum()
    gauss_counts = total_counts * pdf * bin_width

    # --- plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(centers, counts, 'x', label=label_data)
    plt.plot(x, gauss_counts, 'r', linewidth=2, label=label_gauss)

    plt.xlabel("Value")
    plt.ylabel("Counts per bin")
    # plt.ylim(0,12000)
    plt.title("Histogram with ideal Maxwellian overlay, omegatau = {omega_tau0} and c {c_1}")
    plt.xlim(*xlim)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return centers, counts


def abbe_value(x):
    """
    Abbe value:
    A = (1/(n-1)) * sum (x_{i+1}-x_i)^2  /  (2 * var(x))
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.nan
    dif = np.diff(x) #array of length n-1 where each element is the difference between consecutive terms
    num = np.mean(dif**2) #this is mean of the difference square corresponding tot he numerator
    var = np.var(x, ddof=0) #this is half the denominator
    if var == 0:
        return np.nan
    return num / (2.0 * var)


def turning_point_fraction(x):
    """
    Turning point fraction tau:
    tau = (# turning points) / (n-2)
    Turning point at i if (x_i - x_{i-1}) and (x_{i+1} - x_i) have opposite signs.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return np.nan
    d1 = x[1:-1] - x[:-2]
    d2 = x[2:]   - x[1:-1]
    tp = np.sum(d1 * d2 < 0)           # strict turning points
    return tp / (n - 2)


def tarnopolski_path(series, burn_frac=0.2, m_list=None, use_T_over_mu=False):
    """
    Returns arrays A(m), Y(m) where Y is either:
      - tau(m) = turning_point_fraction
      - or T/mu_T  (normalized turning points), if use_T_over_mu=True
    """
    s = np.asarray(series, dtype=float)
    n0 = int(burn_frac * len(s))
    s = s[n0:]  # burn-in removed

    if m_list is None:
        # subsampling scales (like "as subsampling scale is increased")
        m_list = [1,2,3,5,8,12,20,30,40,60,80,100,120,150,170,200]
        # [1,2,3,5,8,12,20,30,50,80,120,200]

    A_vals, Y_vals = [], []
    for m in m_list:
        x = s[::m]
        A = abbe_value(x)
        tau = turning_point_fraction(x)

        if use_T_over_mu:
            # T/mu_T = (3/2)*tau
            Y = 1.5 * tau
        else:
            Y = tau

        A_vals.append(A)
        Y_vals.append(Y)

    return np.array(A_vals), np.array(Y_vals), np.array(m_list)


def plot_T_vs_A(series1, name="u", burn_frac=0.2, m_list=None, use_T_over_mu=False):
    A, Y, m_list = tarnopolski_path(series1, burn_frac=burn_frac, m_list=m_list, use_T_over_mu=use_T_over_mu)
    A_sin, Y_sin, m_list = tarnopolski_path(np.sin(tau), burn_frac=burn_frac, m_list=m_list, use_T_over_mu=use_T_over_mu)
    A_rand, Y_rand, m_list = tarnopolski_path(np.random.randn(N), burn_frac=burn_frac, m_list=m_list, use_T_over_mu=use_T_over_mu)
    
    plt.figure(figsize=(6,5))
    plt.plot(A_sin,Y_sin,label = "sin",color = "green")
    plt.plot(A_rand,Y_rand,label = "rand",color = "blue")
    plt.plot(A, Y,label="series",color = "orange")
    plt.xlabel("Abbe value A")
    plt.ylabel("T/μT" if use_T_over_mu else "τ (turning-point fraction)")
    plt.title(f"Tarnopolski path (subsampling) for {name}, omegatau = {omega_tau0} and q {q}")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    print(Y)

    return A, Y, m_list


def fourier_transform(start=None,end=None,diagnostic = "NO",log = True):
    start = 0 if start is None else start
    end = len(tau) if end is None else end
    t_fft = tau[start:end] #this copies the t array from before
    x_fft = (u[start:end]) #this copies the x array from before
    
    dt_fft = t_fft[1] - t_fft[0] #this calculates the time step
    N_fft  = len(t_fft) #number of samples, basically sets the spacing for omega
    
    # Detrend to remove DC bias (helps visibility of peaks)
    x_fft = x_fft - np.mean(x_fft) #this is done to remove any constants which may give rise to a peak near omega = 0, by doing this you are only focusing on the oscillatory part of the van der pol equation. 
    
    # Hann (Hanning) window and amplitude compensation (coherent gain)
    w = np.hanning(N_fft)                  # Hann window
    w_mean = np.mean(w)                    # = 0.5 for Hann
    xw = x_fft * w #this is bascially to to apply the hanning window, so that xw(t) = x(t)w(t) where w(t) is the Hann window
    
    # Discrete approximation to continuous FT:
    #   \tilde f(ω_k) ≈ (Δt / w_mean) * Σ_n [ x[n] w[n] e^{-i ω_k t_n} ]
    # np.fft.fft computes Σ_n xw[n] e^{-i 2π k n / N}; map k → ω with ω=2π f
    Xw = (dt_fft / w_mean) * np.fft.fft(xw) #the dt_fft basically makes xw continous as np.fft.fft does a discrete sum, also when you are applying the hanning window, it lowers the average amplitude as you are multiplying with w(t), so you essentially divide it again with the mean of w - to restore the original amplitude. 
    
    # Frequency grids: cycles/sec (f) and rad/sec (ω)
    f = np.fft.fftfreq(N_fft, d=dt_fft)          # Hz
    omega = 2.0 * np.pi * f                       # rad/s
    
    # Shift zero freq to center for a two-sided spectrum
    Xw_s   = np.fft.fftshift(Xw)
    omega_s = np.fft.fftshift(omega)
    
    # Magnitude spectrum |tilde f(ω)|
    mag_two_sided = np.abs(Xw_s)
    
    pos = omega_s >= 0
    omega_pos = omega_s[pos]
    mag_pos   = mag_two_sided[pos]
    
    freq_store = omega_pos[::step]
    
    imax = np.argmax(mag_pos)
    center_omega = omega_pos[imax]
    print(f"The first peak is at {center_omega} rad/s")
    
    
    if diagnostic == "NO" and log == True:
        plt.figure(figsize=(9,4))
        plt.semilogy((omega_pos),(mag_pos))
        # plt.xlim(0, 20)  
        # plt.ylim(10e-9,10e3)
        plt.xlabel(r'$\omega$ (rad/s)')
        plt.ylabel(r'$|\tilde f(\omega)|$')
        plt.title('One-sided magnitude spectrum (ω ≥ 0), Hann window, omegatau = {omega_tau0} and q {q}')
        plt.tight_layout()
    
    if diagnostic == "YES" and log == True:
        tau_test = t_fft.copy()
        x_test = np.sin(tau_test)   # angular frequency = 2 rad/s
        
        
        xw_test = x_test * w
        Xw_test = (dt_fft / w_mean) * np.fft.fft(xw_test)
        
        
        Xw_test_s = np.fft.fftshift(Xw_test)
        mag_test_two_sided = np.abs(Xw_test_s)
        mag_test_pos = mag_test_two_sided[pos]
        
        # Locate the test peak (should be ~ 2 rad/s)
        imax_test = np.argmax(mag_test_pos)
        omega_peak_test = omega_pos[imax_test]
        print(f"sin(t) peak ω ≈ {omega_peak_test:.6f} rad/s")
        
        # ---- Overlay plot: VdP vs sin(2 t) on the same one-sided axes ----
        plt.figure(figsize=(9,4))
        plt.semilogy(omega_pos,(mag_pos), label='VdP (one-sided)', linewidth=2)
        plt.semilogy(omega_pos,(mag_test_pos), '--', label='sin(t) (one-sided)', linewidth=2)
        plt.xlim(0, 25)  # shows the fundamental at ~0.3 for VdP and the test spike at 2
        plt.xlabel(r'$\omega$ (rad/s)')
        plt.ylabel(r'$|\tilde f(\omega)|$')
        plt.title('One-sided magnitude spectrum (ω ≥ 0), Hann window — VdP vs sin(t)')
        plt.legend()
        plt.tight_layout()
        
    if diagnostic == "NO" and log == False:
        plt.figure(figsize=(9,4))
        plt.plot(omega_pos,(mag_pos))
        plt.xlim(0, 5)  
        plt.xlabel(r'$\omega$ (rad/s)')
        plt.ylabel(r'$|\tilde f(\omega)|$')
        plt.title('One-sided magnitude spectrum (ω ≥ 0), Hann window')
        plt.tight_layout()
        
    if diagnostic == "YES" and log == False:
        tau_test = t_fft.copy()
        x_test = np.sin(tau_test)   # angular frequency = 2 rad/s
        
        
        xw_test = x_test * w
        Xw_test = (dt_fft / w_mean) * np.fft.fft(xw_test)
        
        
        Xw_test_s = np.fft.fftshift(Xw_test)
        mag_test_two_sided = np.abs(Xw_test_s)
        mag_test_pos = mag_test_two_sided[pos]
        
        # Locate the test peak (should be ~ 2 rad/s)
        imax_test = np.argmax(mag_test_pos)
        omega_peak_test = omega_pos[imax_test]
        print(f"sin(2 t) peak ω ≈ {omega_peak_test:.6f} rad/s")
        
        # ---- Overlay plot: VdP vs sin(2 t) on the same one-sided axes ----
        plt.figure(figsize=(9,4))
        plt.plot(omega_pos, (mag_pos), label='VdP (one-sided)', linewidth=2)
        plt.plot(omega_pos, (mag_test_pos), '--', label='sin(2 t) (one-sided)', linewidth=2)
        # plt.xlim(0, 5)  # shows the fundamental at ~0.3 for VdP and the test spike at 2
        plt.xlabel(r'$\omega$ (rad/s)')
        plt.ylabel(r'$|\tilde f(\omega)|$')
        plt.title('One-sided magnitude spectrum (ω ≥ 0), Hann window — VdP vs sin(2t)')
        plt.legend()
        plt.tight_layout()
        
    # --- choose fit band (YOU tune these) ---
    w1, w2 = 0.1,25  # rad/s example; pick the visually straight region

    mask = (omega_pos >= w1) & (omega_pos <= w2) & (mag_pos > 0)
    x_fit = omega_pos[mask]
    y_fit = np.log(mag_pos[mask]+ 1e-300)
    
    # linear least squares: y = a + b x
    b, a = np.polyfit(x_fit, y_fit, 1)  # returns slope b and intercept a
    
    tau_L = -b  # because b ≈ -tau_L in log|X| vs omega
    print("Fit results:")
    print(f"  slope b = {b:.6e}  (so tau_L = {-b:.6e})")
    print(f"  intercept a = {a:.6e}",end = "\n\n")
    
    omega_line = np.linspace(w1, w2, 400)
    mag_fit_line = np.exp(a + b * omega_line)
    
    # Semilog-y plot with the exponential fit overlaid (paper-style)
    plt.figure(figsize=(9,4))
    plt.semilogy(omega_pos, mag_pos, label=r"$|X(\omega)|$")
    plt.semilogy(omega_line, mag_fit_line, linewidth=3,
                  label=f"fit: |X| = exp(a + b ω)\nτ_L ≈ {tau_L:.3f}")
    plt.xlim(w1,w2)
    plt.xlabel(r'$\omega$ (rad/s)')
    plt.ylabel(r'$|\tilde X(\omega)|$')
    plt.title("Log-linear (semilog-y) spectrum with exponential-fit region, omegatau = {omega_tau0} and q {q}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

        
    return freq_store


def ensemble_fft_SHO_MKT(
    u0_vals,
    variable="u",          # "xi", "u", "Gam", "beta", "th", "ph"
    n_theta=1,
    n_phi=1,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    th0_shift=0.0,
    ph0_shift=0.0,
    burn_frac=0.2,
    # FFT controls
    use_hann=True,
    # plotting / fit controls
    show_plot=True,
    semilog=True,          # if True -> semilog-y plots (for exponential-fit mode)
    loglog=False,          # if True -> log-log plot with TWO fits (paper-style)
    linear=False,          # if True -> normal linear scale plot (no fit)

    # --- Exponential-fit band (used when loglog=False and linear=False) ---
    w1=0.1, w2=20,       # semilog/exponential fit band in rad/s (YOU tune)

    # --- NEW: Two fit bands for loglog=True (paper-style) ---
    pl_w1=1e-2, pl_w2=1e-1,      # power-law fit band (early straight part on log-log)
    exp_w1=1e-1, exp_w2=1.0,     # exponential fit band (later part; fit ln|X| vs ω)

    eps=1e-300
):
    """
    True ensemble FFT:
      - realizations are distinct ICs (theta0, phi0) and distinct u0
      - computes FFT magnitude for each realization, then averages across all

    Modes:

    (1) loglog=False, linear=False:
        - semilog-y plot(s)
        - exponential envelope fit on [w1,w2]:
              ln|X(ω)| ~ a + b ω    =>  |X(ω)| ~ exp(a+bω)
          returns tau_L = -b

    (2) loglog=True (paper-style):
        - log-log plot of the spectrum
        - TWO fits with separate windows you provide:
            A) Power law on [pl_w1,pl_w2]:
                  ln|X| = c + m ln ω  =>  |X| ~ C ω^m
            B) Exponential envelope on [exp_w1,exp_w2]:
                  ln|X| = a + b ω     =>  |X| ~ exp(a+bω)
        - overlays both fits on the same log-log plot
        - returns tau_L from the exponential fit (since that's the meaningful chaotic timescale)

    (3) linear=True:
        - normal linear scale plot of the spectrum (no fit performed)
        - returns omega_pos, mag_mean, None

    Requires globals: t0,t1,dt and your functions:
      SHO_MKT_system_batch(t, Z) and rk4_batch_step(fun, dt, t, Z)
    """

    # --- build theta/phi grid (n_theta*n_phi realizations per u0) ---
    thetas = (np.linspace(0, 2*np.pi, n_theta, endpoint=False) + th0_shift) % (2*np.pi)
    phis   = (np.linspace(0, 2*np.pi, n_phi,   endpoint=False) + ph0_shift) % (2*np.pi)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    th0s = TH.ravel()
    ph0s = PH.ravel()
    M = th0s.size  # = n_theta*n_phi

    # --- time grid ---
    N = int(np.ceil((t1 - t0)/dt)) + 1
    burn_idx = int(burn_frac * N)

    # post-burn segment used for FFT
    Nfft = N - burn_idx
    dt_fft = dt

    # window
    if use_hann:
        w = np.hanning(Nfft)
        w_mean = np.mean(w)
    else:
        w = np.ones(Nfft)
        w_mean = 1.0

    # frequency grid in ω for this Nfft and dt
    f = np.fft.fftfreq(Nfft, d=dt_fft)
    omega = 2.0 * np.pi * f
    omega_s = np.fft.fftshift(omega)
    pos = omega_s >= 0
    omega_pos = omega_s[pos]

    # accumulator for ensemble-averaged magnitude spectrum
    mag_sum = np.zeros_like(omega_pos, dtype=float)
    n_total_realizations = 0

    # map variable name -> column index in Z
    col_map = {"xi":0, "u":1, "Gam":2, "beta":3, "th":4, "ph":5}
    if variable not in col_map:
        raise ValueError(f"variable must be one of {list(col_map.keys())}")
    j = col_map[variable]

    # --- loop over velocities; each loop runs M trajectories in parallel ---
    for u0 in u0_vals:
        # initial batch state (M,6)
        Zb = np.zeros((M, 6), dtype=float)
        Zb[:, 0] = xi0
        Zb[:, 1] = u0
        Zb[:, 2] = Gam0
        Zb[:, 3] = beta0
        Zb[:, 4] = th0s
        Zb[:, 5] = ph0s

        # store variable post-burn only: shape (Nfft, M)
        X = np.empty((Nfft, M), dtype=float)

        t = t0
        write_ptr = 0
        for k in range(N):
            if k >= burn_idx:
                X[write_ptr, :] = Zb[:, j]
                write_ptr += 1

            if k < N - 1:
                Zb = rk4_batch_step(SHO_MKT_system_batch, dt, t, Zb)
                t += dt

        # detrend per realization (remove DC)
        X = X - X.mean(axis=0, keepdims=True)

        # apply window and coherent gain correction
        Xw = X * w[:, None]

        # FFT along time axis for all realizations at once
        F = (dt_fft / w_mean) * np.fft.fft(Xw, axis=0)
        Fs = np.fft.fftshift(F, axes=0)
        mag_two_sided = np.abs(Fs)

        # one-sided ω>=0
        mag_pos = mag_two_sided[pos, :]     # shape (n_freq_pos, M)

        # average magnitude over realizations in THIS batch, then accumulate
        mag_batch_mean = mag_pos.mean(axis=1)  # shape (n_freq_pos,)
        mag_sum += mag_batch_mean * M
        n_total_realizations += M

    # final ensemble average over ALL realizations
    mag_mean = mag_sum / n_total_realizations

    # peak (optional)
    imax = np.argmax(mag_mean)
    print(f"[ensemble avg] First peak at ω = {omega_pos[imax]} rad/s")

    # ===========================
    # MODE: LINEAR (normal scale, no fit)
    # ===========================
    if linear:
        if show_plot:
            plt.figure(figsize=(9,4))
            plt.plot(omega_pos, mag_mean, label=r"$\langle |X(\omega)| \rangle$")
            plt.xlim(w1, w2) 
            plt.xlabel(r"$\omega$ (rad/s)")
            plt.ylabel(r"$|\tilde X(\omega)|$")
            plt.title(f"Ensemble-averaged spectrum (linear scale) for {variable}, omegatau = {omega_tau0} and q {q}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return omega_pos, mag_mean, None

    # ===========================
    # MODE: LOG-LOG with TWO fits (paper-style)
    # ===========================
    if loglog:
        # plot full spectrum on log-log
        if show_plot:
            plt.figure(figsize=(9,4))
            mask_plot = (omega_pos > 0) & (mag_mean > 0)
            plt.loglog(omega_pos[mask_plot], mag_mean[mask_plot],
                        label=r"$\langle |X(\omega)| \rangle$")
            plt.xlim(1,50)
            plt.xlabel(r"$\omega$ (rad/s)")
            plt.ylabel(r"$|\tilde X(\omega)|$")
            plt.title(f"Ensemble-averaged spectrum (full range, log-log) for {variable}, omegatau = {omega_tau0} and q {q}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # ---- (A) POWER-LAW FIT on [pl_w1, pl_w2] ----
        pl_mask = (omega_pos >= pl_w1) & (omega_pos <= pl_w2) & (omega_pos > 0) & (mag_mean > 0)
        w_pl = omega_pos[pl_mask]
        y_pl = mag_mean[pl_mask]

        logw = np.log(w_pl + eps)
        logy = np.log(y_pl + eps)
        m, c = np.polyfit(logw, logy, 1)   # logy = c + m logw
        C = np.exp(c)

        print("Power-law fit (log-log):")
        print(f"  band [{pl_w1},{pl_w2}]")
        print(f"  m = {m:.6e}   so |X| ~ C ω^m")
        print(f"  C = {C:.6e}", end="\n\n")

        w_pl_line = np.linspace(pl_w1, pl_w2, 400)
        y_pl_line = C * (w_pl_line**m)

        # ---- (B) EXPONENTIAL ENVELOPE FIT on [exp_w1, exp_w2] ----
        exp_mask = (omega_pos >= exp_w1) & (omega_pos <= exp_w2) & (mag_mean > 0)
        w_exp = omega_pos[exp_mask]
        y_exp = mag_mean[exp_mask]

        # ln|X| = a + b ω
        b, a = np.polyfit(w_exp, np.log(y_exp + eps), 1)
        tau_L = -b

        print("Exponential-envelope fit (on log-linear in ω):")
        print(f"  band [{exp_w1},{exp_w2}]")
        print(f"  slope b = {b:.6e}  -> tau_L = {-b:.6e}")
        print(f"  intercept a = {a:.6e}", end="\n\n")

        w_exp_line = np.linspace(exp_w1, exp_w2, 400)
        y_exp_line = np.exp(a + b*w_exp_line)

        # overlay both fits on the SAME log-log plot
        if show_plot:
            plt.figure(figsize=(9,4))
            plt.loglog(omega_pos[mask_plot], mag_mean[mask_plot],
                   label=r"$\langle |X(\omega)| \rangle$")
            plt.loglog(w_pl_line, y_pl_line, linewidth=3,
                        label=f"power-law fit [{pl_w1},{pl_w2}]: m≈{m:.3g}")
            plt.loglog(w_exp_line, y_exp_line, linewidth=3,
                        label=f"exp envelope fit [{exp_w1},{exp_w2}]: τ_L≈{tau_L:.3g}")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return omega_pos, mag_mean, tau_L

    # ===========================
    # MODE: SEMILOG-Y exponential fit (paper Fig.4 style)
    # (Removed the ln(mag) plotting branch as you requested)
    # ===========================
    if show_plot:
        plt.figure(figsize=(9,4))
        plt.semilogy(omega_pos, mag_mean, label=r"$\langle |X(\omega)| \rangle$")
        plt.xlabel(r"$\omega$ (rad/s)")
        plt.ylabel(r"$|\tilde X(\omega)|$")
        plt.title(f"Ensemble-averaged spectrum (full range, semilog-y) for {variable},omegatau = {omega_tau0} and q {q}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- exponential (paper-style) fit: ln|X| = a + b ω ---
    fitmask = (omega_pos >= w1) & (omega_pos <= w2) & (mag_mean > 0)
    x_fit = omega_pos[fitmask]
    y_fit = np.log(mag_mean[fitmask] + eps)

    b, a = np.polyfit(x_fit, y_fit, 1)
    tau_L = -b
    print("Fit results (ensemble avg):")
    print(f"  slope b = {b:.6e}  -> tau_L = {-b:.6e}")
    print(f"  intercept a = {a:.6e}", end="\n\n")

    omega_line = np.linspace(w1, w2, 400)
    mag_fit_line = np.exp(a + b * omega_line)

    if show_plot:
        plt.figure(figsize=(9,4))
        plt.semilogy(omega_pos, mag_mean, label=r"$\langle |X(\omega)| \rangle$")
        plt.xlim(w1, w2)
        plt.xlabel(r"$\omega$ (rad/s)")
        plt.ylabel(r"$|\tilde X(\omega)|$")
        plt.title(f"Ensemble-averaged spectrum with exponential fit (semilog-y) for {variable},omegatau = {omega_tau0} and q {q}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return omega_pos, mag_mean, tau_L


def lorentzian_width_from_timeseries(t, x, tmin=None, tmax=None, use_abs=True):
    """
    Estimate Lorentzian width from a prominent pulse in x(t).
    Returns: t_peak, fwhm, tauL_time (=fwhm/2 for Lorentzian A/(1+((t-t0)/tau)^2)).
    """
    # restrict to window if provided
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= (t >= tmin)
    if tmax is not None:
        mask &= (t <= tmax)

    tt = t[mask]
    xx = x[mask]

    # choose pulse based on |x| or x
    y = np.abs(xx) if use_abs else xx

    i0 = np.argmax(y)
    t0 = tt[i0]
    peak = y[i0]
    half = 0.5 * peak

    # walk left to half max
    il = i0
    while il > 0 and y[il] > half:
        il -= 1
    # walk right to half max
    ir = i0
    while ir < len(y)-1 and y[ir] > half:
        ir += 1

    if il == 0 or ir == len(y)-1:
        return t0, np.nan, np.nan

    # linear interpolation for half crossings
    tL1, tL2 = tt[il], tt[il+1]
    yL1, yL2 = y[il], y[il+1]
    t_left = tL1 + (half - yL1) * (tL2 - tL1) / (yL2 - yL1)

    tR1, tR2 = tt[ir-1], tt[ir]
    yR1, yR2 = y[ir-1], y[ir]
    t_right = tR1 + (half - yR1) * (tR2 - tR1) / (yR2 - yR1)

    fwhm = t_right - t_left
    tauL_time = fwhm / 2.0
    return t0, fwhm, tauL_time

def plot_pulse_with_lorentzian_overlay(t, x, t_center=None, window=20.0, use_abs=True):
    if t_center is None:
        # choose center based on global max excursion
        t_center = t[np.argmax(np.abs(x))] if use_abs else t[np.argmax(x)]

    tmin = t_center - window/2
    tmax = t_center + window/2

    # restrict to window
    mask = (t >= tmin) & (t <= tmax)
    tt = t[mask]
    xx = x[mask]

    # choose peak index inside window
    if use_abs:
        i0 = np.argmax(np.abs(xx))
    else:
        i0 = np.argmax(xx)

    t0 = tt[i0]
    peak_val = xx[i0]                 # signed peak value
    peak_mag = abs(peak_val)
    half = 0.5 * peak_mag

    # for FWHM we work with magnitude if use_abs
    y = np.abs(xx) if use_abs else xx

    # walk left/right to find half-maximum crossings
    il = i0
    while il > 0 and (abs(y[il]) if use_abs else y[il]) > half:
        il -= 1

    ir = i0
    while ir < len(y)-1 and (abs(y[ir]) if use_abs else y[ir]) > half:
        ir += 1

    if il == 0 or ir == len(y)-1:
        print("Pulse not fully contained in window; increase window.")
        return np.nan

    # interpolate left crossing
    tL1, tL2 = tt[il], tt[il+1]
    yL1, yL2 = (abs(y[il]) if use_abs else y[il]), (abs(y[il+1]) if use_abs else y[il+1])
    t_left = tL1 + (half - yL1) * (tL2 - tL1) / (yL2 - yL1)

    # interpolate right crossing
    tR1, tR2 = tt[ir-1], tt[ir]
    yR1, yR2 = (abs(y[ir-1]) if use_abs else y[ir-1]), (abs(y[ir]) if use_abs else y[ir])
    t_right = tR1 + (half - yR1) * (tR2 - tR1) / (yR2 - yR1)

    fwhm = t_right - t_left
    tauL_time = fwhm / 2.0

    # signed Lorentzian overlay
    lor = peak_val / (1.0 + ((tt - t0)/tauL_time)**2)

    plt.figure(figsize=(7,4))
    plt.plot(tt, xx, label="x(tau) segment")
    plt.plot(tt, lor, 'r', linewidth=2, label=f"Lorentzian fit, tau_L={tauL_time:.3g}")
    plt.xlabel("tau")
    plt.ylabel("x")
    plt.title("Time-domain pulse and signed Lorentzian overlay,omegatau = {omega_tau0} and q {q}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("t_peak =", t0)
    print("peak_val =", peak_val)
    print("FWHM =", fwhm)
    print("tau_L(time) =", tauL_time)
    return tauL_time

def ensemble_running_time_averages_x2_v2(
    u0_vals,
    n_theta=20,
    n_phi=20,
    xi0=0.0,
    Gam0=0.0,
    beta0=0.0,
    th0_shift=0.0,
    ph0_shift=0.0,
    burn_frac=0.2,
    store_step=10,          # store every this many dt steps (keep moderate for memory)
    show_plot=True
):
    """
    Computes ensemble average as a function of time of the RUNNING time averages:
        overline_x2^(r)(t) = (1/(t-t_burn)) ∫ x_r(s)^2 ds
        overline_v2^(r)(t) = (1/(t-t_burn)) ∫ v_r(s)^2 ds

    Realizations = (theta0,phi0) grid for each u0 in u0_vals.

    Returns:
        t_store
        mean_x2bar(t), mean_v2bar(t)
        msd_x2bar(t),  msd_v2bar(t)   (across realizations)
    """

    # --- theta/phi grid ---
    thetas = (np.linspace(0, 2*np.pi, n_theta, endpoint=False) + th0_shift) % (2*np.pi)
    phis   = (np.linspace(0, 2*np.pi, n_phi,   endpoint=False) + ph0_shift) % (2*np.pi)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    th0s = TH.ravel()
    ph0s = PH.ravel()
    M = th0s.size

    # --- time grid ---
    N = int(np.ceil((t1 - t0)/dt)) + 1
    burn_idx = int(burn_frac * N)
    t_burn = t0 + burn_idx * dt

    # store times AFTER burn
    post_N = N - burn_idx
    post_indices = np.arange(0, post_N, store_step)  # indices in post-burn coordinates
    n_store = len(post_indices)
    t_store = t_burn + post_indices * dt

    # We'll accumulate sums over ALL realizations (across all u0 and theta/phi)
    # mean of running-avg curves:
    sum_x2bar = np.zeros(n_store, dtype=float)
    sum_v2bar = np.zeros(n_store, dtype=float)
    sum_Gambar  = np.zeros(n_store, dtype=float)
    # second moment for MSD:
    sum_x2bar2 = np.zeros(n_store, dtype=float)
    sum_v2bar2 = np.zeros(n_store, dtype=float)
    sum_Gambar2 = np.zeros(n_store, dtype=float)

    R_total = 0  # total number of realizations accumulated

    for u0 in u0_vals:
        # initial batch state
        Zb = np.zeros((M, 6), dtype=float)
        Zb[:, 0] = xi0
        Zb[:, 1] = u0
        Zb[:, 2] = Gam0
        Zb[:, 3] = beta0
        Zb[:, 4] = th0s
        Zb[:, 5] = ph0s

        # Running integrals AFTER burn: I_x2(t) = ∫ x^2 dt, I_v2(t)=∫ v^2 dt
        I_x2 = np.zeros(M, dtype=float)
        I_v2 = np.zeros(M, dtype=float)
        I_Gam = np.zeros(M, dtype=float)

        # We'll store running averages at the requested stored times:
        x2bar_store = np.zeros((n_store, M), dtype=float)
        v2bar_store = np.zeros((n_store, M), dtype=float)
        Gambar_store = np.zeros((n_store, M), dtype=float)

        t = t0
        post_k = 0              # counts steps after burn
        store_ptr = 0

        for k in range(N):
            if k >= burn_idx:
                # accumulate integrals using rectangle rule on [t, t+dt)
                # (skip at the very first post-burn sample if you want; effect is tiny for long runs)
                x = Zb[:, 0]
                v = Zb[:, 1]
                Gam = Zb[:, 2] 
                
                I_x2 += (x*x) * dt
                I_v2 += (v*v) * dt
                I_Gam += Gam   * dt 

                # if this post-burn index is in our store list, record running averages
                if store_ptr < n_store and post_k == post_indices[store_ptr]:
                    elapsed = (post_k + 1) * dt  # time since burn included in integral
                    x2bar_store[store_ptr, :] = I_x2 / elapsed
                    v2bar_store[store_ptr, :] = I_v2 / elapsed
                    Gambar_store[store_ptr, :] = I_Gam / elapsed
                    store_ptr += 1

                post_k += 1

            if k < N - 1:
                Zb = rk4_batch_step(SHO_MKT_system_batch, dt, t, Zb)
                t += dt

        # Now we have running-avg curves for M realizations at n_store times
        # Compute ensemble contributions:
        x2bar_mean_batch = x2bar_store.mean(axis=1)      # shape (n_store,)
        v2bar_mean_batch = v2bar_store.mean(axis=1)
        

        x2bar_second_batch = (x2bar_store**2).mean(axis=1)
        v2bar_second_batch = (v2bar_store**2).mean(axis=1)

        sum_x2bar += x2bar_mean_batch * M
        sum_v2bar += v2bar_mean_batch * M
        sum_Gambar  += Gambar_store.mean(axis=1) * M
        sum_x2bar2 += x2bar_second_batch * M
        sum_v2bar2 += v2bar_second_batch * M

        R_total += M

    # Ensemble mean curves
    mean_x2bar = sum_x2bar / R_total
    mean_v2bar = sum_v2bar / R_total
    mean_Gambar = sum_Gambar / R_total

    # Ensemble MSD across realizations: E[Y^2] - (E[Y])^2
    msd_x2bar = (sum_x2bar2 / R_total) - mean_x2bar**2
    msd_v2bar = (sum_v2bar2 / R_total) - mean_v2bar**2

    if show_plot:
        plt.figure(figsize=(9,4))
        plt.plot(t_store, mean_x2bar, label=r"$\langle \overline{x^2}(t)\rangle$")
        plt.plot(t_store, mean_v2bar, label=r"$\langle \overline{v^2}(t)\rangle$")
        plt.plot(t_store, mean_x2bar/mean_v2bar, label=r"$\langle \overline{ratio}(t)\rangle$")
        plt.plot(t_store, mean_Gambar, label=r"$\langle \overline{\Gamma}(t)\rangle$")  # NEW
        plt.axhline(q, ls='--', color='gray',
                    label=f'q = {q} (expected $\\langle\\Gamma\\rangle$)')  
        plt.xlabel(r"$\tau$")
        plt.ylabel("running time average")
        plt.title(f"Ensemble mean of running time-averaged $x^2$ and $v^2$,omegatau = {omega_tau0} and q {q}")
        plt.legend()
        plt.tight_layout()
        plt.show()


    return t_store, mean_x2bar, mean_v2bar, mean_Gambar


def ensemble_running_kurtosis_v(
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
    eps=1e-15,   # prevents divide-by-zero if something weird happens
):

    # --- theta/phi grid ---
    thetas = (np.linspace(0, 2*np.pi, n_theta, endpoint=False) + th0_shift) % (2*np.pi)
    phis   = (np.linspace(0, 2*np.pi, n_phi,   endpoint=False) + ph0_shift) % (2*np.pi)
    TH, PH = np.meshgrid(thetas, phis, indexing="ij")
    th0s = TH.ravel()
    ph0s = PH.ravel()
    M = th0s.size

    # --- time grid ---
    N = int(np.ceil((t1 - t0)/dt)) + 1
    burn_idx = int(burn_frac * N)
    t_burn = t0 + burn_idx * dt

    post_N = N - burn_idx
    post_indices = np.arange(0, post_N, store_step)
    n_store = len(post_indices)
    t_store = t_burn + post_indices * dt

    # accumulate across all realizations (all u0 and all theta/phi)
    sum_K = np.zeros(n_store, dtype=float)
    sum_v2bar = np.zeros(n_store, dtype=float)
    sum_v4bar = np.zeros(n_store, dtype=float)

    R_total = 0

    for u0 in u0_vals:
        # initial batch state
        Zb = np.zeros((M, 6), dtype=float)
        Zb[:, 0] = xi0
        Zb[:, 1] = u0
        Zb[:, 2] = Gam0
        Zb[:, 3] = beta0
        Zb[:, 4] = th0s
        Zb[:, 5] = ph0s

        # running integrals after burn
        I_v2 = np.zeros(M, dtype=float)
        I_v4 = np.zeros(M, dtype=float)

        
        v2bar_store = np.zeros((n_store, M), dtype=float)
        v4bar_store = np.zeros((n_store, M), dtype=float)

        t = t0
        post_k = 0
        store_ptr = 0

        for k in range(N):
            if k >= burn_idx:
                v = Zb[:, 1]
                vv = v*v
                I_v2 += vv * dt
                I_v4 += (vv*vv) * dt   # v^4 * dt

                if store_ptr < n_store and post_k == post_indices[store_ptr]:
                    elapsed = (post_k + 1) * dt
                    v2bar_store[store_ptr, :] = I_v2 / elapsed
                    v4bar_store[store_ptr, :] = I_v4 / elapsed
                    store_ptr += 1

                post_k += 1

            if k < N - 1:
                Zb = rk4_batch_step(SHO_MKT_system_batch, dt, t, Zb)
                t += dt

        # ensemble contributions from this u0 batch
        K_store = v4bar_store / (v2bar_store**2 + eps)  # (n_store, M)

        sum_K     += K_store.mean(axis=1) * M
        sum_v2bar += v2bar_store.mean(axis=1) * M
        sum_v4bar += v4bar_store.mean(axis=1) * M
        R_total += M

        # sum_v2bar += v2bar_store.mean(axis=1) * M
        # sum_v4bar += v4bar_store.mean(axis=1) * M
        # R_total += M

    
    mean_K     = sum_K / R_total      # <-- correct kurtosis
    mean_v2bar = sum_v2bar / R_total
    mean_v4bar = sum_v4bar / R_total
    
    # mean_v2bar = sum_v2bar / R_total
    # mean_v4bar = sum_v4bar / R_total

    # K_curve = mean_v4bar / (mean_v2bar**2)

    if show_plot:
        plt.figure(figsize=(9,4))
        plt.plot(t_store, mean_K, label=r"$K(\tau)=\langle \overline{v^4}\rangle / \langle \overline{v^2}\rangle^2$")
        plt.xlabel(r"$\tau$")
        plt.ylabel("kurtosis")
        plt.title(f"Ensemble running kurtosis of velocity, omegatau = {omega_tau0} and q {q}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return t_store, mean_K, mean_v2bar, mean_v4bar

# def dfa_hurst(x, min_win=16, max_win=None, n_scales=20, burn_frac=0.2):
#     """
#     Detrended Fluctuation Analysis.
#     Returns H (Hurst exponent) and the fit quality.
    
#     H ~ 0.5  -> chaotic/random
#     H ~ 1.0  -> periodic/limit cycle
#     """
#     x = np.asarray(x, dtype=float)
#     # burn-in removal
#     x = x[int(burn_frac * len(x)):]
#     x = x - np.mean(x)
    
#     # cumulative sum profile
#     Y = np.cumsum(x)
#     N = len(Y)
    
#     if max_win is None:
#         max_win = N // 4
    
#     scales = np.unique(np.round(
#         np.logspace(np.log10(min_win), np.log10(max_win), n_scales)
#     ).astype(int))
    
#     F_vals = []
#     valid_scales = []
    
#     for s in scales:
#         if s < 4 or s > N // 2:
#             continue
        
#         n_segs = N // s
#         if n_segs < 2:
#             continue
        
#         rms_list = []
#         for i in range(n_segs):
#             seg = Y[i*s:(i+1)*s]
#             t_seg = np.arange(s, dtype=float)
            
#             # detrend: fit and subtract linear trend
#             coeffs = np.polyfit(t_seg, seg, 1)
#             trend  = np.polyval(coeffs, t_seg)
#             rms_list.append(np.sqrt(np.mean((seg - trend)**2)))
        
#         F_vals.append(np.mean(rms_list))
#         valid_scales.append(s)
    
#     valid_scales = np.array(valid_scales, dtype=float)
#     F_vals       = np.array(F_vals,       dtype=float)
    
#     # H from slope of log F vs log s
#     log_s = np.log(valid_scales)
#     log_F = np.log(F_vals)
#     H, intercept = np.polyfit(log_s, log_F, 1)
    
#     return H, valid_scales, F_vals


# def hurst_vs_q(
#     q_vals,
#     omega_tau0_val=5,
#     variable="u",
#     t0_=0.0, t1_=1200.0, dt_=0.01,
#     f0_=0.18,
#     burn_frac=0.2,
#     show_dfa_plots=False   # set True to see individual DFA fits
# ):
#     """
#     For each q in q_vals, run the MKT-SHO system and compute
#     Hurst exponent H via DFA. Plots H vs q and marks critical q ~ 1/omega_tau.
#     """
#     H_vals = []
    
#     N_ = int(np.ceil((t1_ - t0_) / dt_)) + 1
#     tau_ = np.linspace(t0_, t1_, N_)
    
#     col_map = {"xi":0, "u":1, "Gam":2, "beta":3, "th":4, "ph":5}
#     j = col_map[variable]
    
#     for q_ in q_vals:
#         print(f"Running q = {q_:.4f} ...")
        
#         def mkt_rhs(tau, z):
#             xi_, u_, G_, b_, th_, ph_ = z
#             xi_dot  = u_
#             u_dot   = -xi_ + f0_*(np.sin(th_) + np.sin(ph_)) - G_*u_ + q_*u_
#             G_dot   = (1/omega_tau0_val**2)*(u_**2 - 1.0) - b_*G_
#             b_dot   = G_**2 - (1/omega_tau0_val**2)
#             th_dot  = G_
#             ph_dot  = b_
#             return np.array([xi_dot, u_dot, G_dot, b_dot, th_dot, ph_dot])
        
#         # integrate
#         Z_ = np.zeros((N_, 6), dtype=float)
#         Z_[0] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        
#         for k in range(N_ - 1):
#             Z_[k+1] = rk4singlestep(mkt_rhs, dt_, tau_[k], Z_[k])
            
#             # early exit if diverging
#             if np.any(np.abs(Z_[k+1]) > 1e6):
#                 print(f"  -> DIVERGED at tau={tau_[k+1]:.1f}")
#                 Z_[k+1:] = np.nan
#                 break
        
#         series = Z_[:, j]
        
#         # if diverged, assign H=nan
#         if np.any(np.isnan(series)):
#             H_vals.append(np.nan)
#             continue
        
#         H, scales, F_s = dfa_hurst(series, burn_frac=burn_frac)
#         H_vals.append(H)
        
#         if show_dfa_plots:
#             plt.figure(figsize=(5, 3))
#             plt.loglog(scales, F_s, 'o-')
#             plt.loglog(scales, np.exp(np.log(scales)*H + 
#                        (np.log(F_s).mean() - H*np.log(scales).mean())),
#                        'r--', label=f'H={H:.3f}')
#             plt.xlabel('scale s'); plt.ylabel('F(s)')
#             plt.title(f'DFA: q={q_:.3f}, ω_τ={omega_tau0_val}')
#             plt.legend(); plt.tight_layout(); plt.show()
    
#     H_vals = np.array(H_vals)
#     q_c = 1.0 / omega_tau0_val
    
#     # --- PLOT ---
#     plt.figure(figsize=(8, 4))
#     plt.plot(q_vals, H_vals, 'o-', color='steelblue', label=f'H({variable})')
#     plt.axvline(q_c, ls='--', color='red',
#                 label=f'$q_c = 1/\\omega_\\tau = {q_c:.3f}$')
#     plt.axhline(0.5, ls=':', color='gray', label='H=0.5 (random/chaotic)')
#     plt.axhline(1.0, ls=':', color='green', label='H=1.0 (periodic)')
#     plt.xlabel('q')
#     plt.ylabel('Hurst exponent H')
#     plt.title(f'Hurst exponent vs q  (ω_τ={omega_tau0_val}, variable={variable})')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    
#     # print the transition region
#     transition_mask = (H_vals > 0.6) & (H_vals < 0.9)
#     if np.any(transition_mask):
#         print(f"\nTransition region (0.6 < H < 0.9): "
#               f"q ∈ [{q_vals[transition_mask].min():.3f}, "
#               f"{q_vals[transition_mask].max():.3f}]")
    
#     return q_vals, H_vals


# --- Usage ---
# Fine scan around q_c = 1/omega_tau0 = 0.2 for omega_tau0=5
# q_scan = np.linspace(0.005, 2, 10)
# q_out, H_out = hurst_vs_q(
#     q_vals=q_scan,
#     omega_tau0_val=5,
#     variable="u",
#     burn_frac=0.2
# )





# # # ---- 2) tau_L from Lorentzian pulse width in time series (Fig 5 style) ----
# tauL_time = plot_pulse_with_lorentzian_overlay(tau, u, window=20.0)

# print("tau_L from time pulse     =", tauL_time)



# for semilog
# omega_pos, mag_mean, tau_L = ensemble_fft_SHO_MKT(
#     u0_vals=np.linspace(-5, 5, 21),
#     variable="u",
#     loglog=False,
#     semilog=True,
#     burn_frac=0.2,
#     w1=0.1, w2=25
# )

# # # # for linear
# omega_pos, mag_mean, tau_L = ensemble_fft_SHO_MKT(
#     u0_vals=np.linspace(-5, 5, 21),
#     variable="u",
#     loglog=False,
#     semilog=False,
#     linear=True,
#     burn_frac=0.2,
#     w1=0.1, w2=5
# )



# for log-log
# omega_pos, mag_mean, tau_L = ensemble_fft_SHO_MKT(
#     u0_vals=np.linspace(-5, 5, 1),
#     variable="xi",
#     loglog=True,
#     burn_frac=0.2,

#     # power-law region (straight early part on log-log)
#     pl_w1=10**(-2.2), pl_w2=10**(1),

#     # exponential-envelope region (later tail)
#     exp_w1=1, exp_w2=10**(2.8)
# )


   
# xi_plot()
u_plot()
# # # q_minus_gamma_plot()
# phase_portrait(500,2900)
# Gam_u_series_plot() 
#  gam_plot()
# phase_portrait_gam_u(200,1200)

# start_idx = np.searchsorted(tau, 200) 

# fourier_transform(start = start_idx,diagnostic="NO",log=False) #diagnostic



# u0_vals = np.linspace(-5,5, 21)  #if you want the scaled xi distribution use the code in the other page
# plot_ensemble_avg_with_ideal_gaussian(
#     u0_vals,
#     variable="u",
#     bins=41,
#     xlim=(-4,4),
#     n_theta=20,
#     n_phi=20,
#     burn_frac=0.2,
#     store_step=50,
#     gaussian_mode="standard",  # N(0,1)
#     xlabel="u",
# )

# t_store, mean_x2bar, mean_v2bar, mean_Gambar = ensemble_running_time_averages_x2_v2(
#     u0_vals=u0_vals,
#     n_theta=20,
#     n_phi=20,
#     burn_frac=0.2,
#     store_step=10,   # you can increase (e.g., 50) if you want fewer time points
#     show_plot=True
# )

# print(mean_v2bar)
# print(mean_x2bar)
# print(mean_Gambar)


# t_store, mean_K, mean_v2bar, mean_v4bar = ensemble_running_kurtosis_v(
#     u0_vals=u0_vals,
#     n_theta=20,
#     n_phi=20,
#     burn_frac=0.2, 
#     # this was intitially 0.2
#     store_step=10,
#     show_plot=True
# )

# print("Final (late-time) kurtosis estimate:", mean_K[-1])

# print(np.sqrt(2*(1+np.mean(beta)*np.mean(Gam)*(omega_tau0)**2)))


# xi, u, theta, phi, t are arrays





# plot_data_with_ideal_maxwellian(u) #only for one intial condition



# def hurst_RS(x, min_win=16, max_win=None, n_scales=20, burn_frac=0.2):
#     """
#     Classical Hurst exponent via R/S analysis. Bounded between 0 and 1.
#     H = 0.5  -> chaotic/random
#     H -> 1.0 -> periodic limit cycle
#     """
#     x = np.asarray(x, dtype=float)
#     x = x[int(burn_frac * len(x)):]
#     x = x - np.mean(x)
#     N = len(x)

#     if max_win is None:
#         max_win = N // 4

#     scales = np.unique(np.round(
#         np.logspace(np.log10(min_win), np.log10(max_win), n_scales)
#     ).astype(int))

#     RS_vals = []
#     valid_scales = []

#     for s in scales:
#         if s < 4 or s > N // 2:
#             continue
#         n_segs = N // s
#         if n_segs < 2:
#             continue

#         rs_list = []
#         for i in range(n_segs):
#             seg = x[i*s:(i+1)*s]
#             seg = seg - np.mean(seg)
#             Z_seg = np.cumsum(seg)
#             R = np.max(Z_seg) - np.min(Z_seg)
#             S = np.std(seg, ddof=1)
#             if S > 0:
#                 rs_list.append(R / S)

#         if len(rs_list) > 0:
#             RS_vals.append(np.mean(rs_list))
#             valid_scales.append(s)

#     if len(valid_scales) < 2:
#         return np.nan, np.array([]), np.array([])

#     valid_scales = np.array(valid_scales, dtype=float)
#     RS_vals      = np.array(RS_vals,      dtype=float)
#     H, _         = np.polyfit(np.log(valid_scales), np.log(RS_vals), 1)

#     return H, valid_scales, RS_vals


# def hurst_vs_q(q_vals, variable="u", burn_frac=0.2):
#     """
#     Scans q values using the already-integrated global arrays.
#     Change omega_tau0 and re-run the integration at the top of the
#     script, then call this function.
#     """
#     col_map = {"xi": 0, "u": 1, "Gam": 2, "beta": 3, "th": 4, "ph": 5}
#     j = col_map[variable]
#     q_c = 1.0 / omega_tau0

#     H_vals = []

#     for q_ in q_vals:
#         print(f"Running q = {q_:.4f} ...")

#         # integrate for this q using existing global setup
#         def mkt_rhs(tau, z):
#             xi_, u_, G_, b_, th_, ph_ = z
#             xi_dot  = u_
#             u_dot   = -xi_ + f0*(np.sin(th_) + np.sin(ph_)) - G_*u_ + q_*u_
#             G_dot   = (1/omega_tau0**2)*(u_**2 - 1.0) - b_*G_
#             b_dot   = G_**2 - (1/omega_tau0**2)
#             th_dot  = G_
#             ph_dot  = b_
#             return np.array([xi_dot, u_dot, G_dot, b_dot, th_dot, ph_dot])

#         Z_ = np.zeros((N, 6), dtype=float)
#         Z_[0] = z0.copy()

#         diverged = False
#         for k in range(N - 1):
#             Z_[k+1] = rk4singlestep(mkt_rhs, dt, tau[k], Z_[k])
#             if np.any(np.abs(Z_[k+1]) > 1e6):
#                 print(f"  -> DIVERGED at tau={tau[k+1]:.1f}")
#                 diverged = True
#                 break

#         if diverged:
#             H_vals.append(np.nan)
#             continue

#         series = Z_[:, j]
#         H, _, _ = hurst_RS(series, burn_frac=burn_frac)
#         print(f"  H = {H:.4f}")
#         H_vals.append(H)

#     H_vals = np.array(H_vals)

#     plt.figure(figsize=(9, 5))
#     plt.plot(q_vals, H_vals, 'o-', color='steelblue', label=f'H({variable})')
#     plt.axvline(q_c, ls='--', color='red',
#                 label=f'$q_c = 1/\\omega_\\tau = {q_c:.3f}$')
#     plt.axhline(0.5, ls=':', color='gray',  label='H=0.5 (chaotic)')
#     plt.axhline(1.0, ls=':', color='green', label='H=1.0 (periodic)')
#     plt.ylim(0, 1.15)
#     plt.xlabel('q')
#     plt.ylabel('Hurst exponent H')
#     plt.title(f'Hurst exponent vs q, omegatau = {omega_tau0}')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     return q_vals, H_vals


# # ========= USAGE =========
# q_scan = np.linspace(0, 0.7, 20)  # adjust range for your omega_tau0
# q_out, H_out = hurst_vs_q(q_scan, variable="u", burn_frac=0.2)




















































