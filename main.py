#!/usr/bin/env python3
"""Command-line entry point for the Van der Pol + MKT thermostat analyses.

Examples
--------
    python main.py trajectory --plot u
    python main.py histogram --save figures/
    python main.py fft --mode loglog
    python main.py kurtosis --t1 400          # shorter run for a quick look

Run `python main.py <command> --help` for all options.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermostat import (
    Params,
    ensemble_fft,
    ensemble_running_kurtosis_v,
    ensemble_running_time_averages_x2_v2,
    fourier_transform,
    gam_plot,
    gam_u_series_plot,
    integrate,
    phase_portrait,
    phase_portrait_gam_u,
    plot_ensemble_avg_with_ideal_gaussian,
    plot_pulse_with_lorentzian_overlay,
    q_minus_gamma_plot,
    u_plot,
    xi_plot,
)

TRAJECTORY_PLOTS = {
    "u": u_plot,
    "xi": xi_plot,
    "gam": gam_plot,
    "q-gam": q_minus_gamma_plot,
    "gam-u": gam_u_series_plot,
}
PHASE_PLOTS = {"phase": phase_portrait, "phase-gam-u": phase_portrait_gam_u}


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--omega-tau", type=float, default=10.0, help="thermostat timescale omega_tau0 (default 10)")
    common.add_argument("--q", type=float, default=0.5, help="parameter q (default 0.5)")
    common.add_argument("--alpha", type=float, default=7.0, help="Van der Pol strength (default 7)")
    common.add_argument("--f0", type=float, default=0.2, help="phase-drive amplitude (default 0.2)")
    common.add_argument("--t1", type=float, default=1200.0, help="end of integration in tau (default 1200)")
    common.add_argument("--dt", type=float, default=0.01, help="RK4 step (default 0.01)")
    common.add_argument("--save", metavar="DIR", help="save figures as PNGs in DIR instead of showing them")

    ensemble = argparse.ArgumentParser(add_help=False)
    ensemble.add_argument("--n-u0", type=int, default=21, help="number of initial velocities in [-5, 5] (default 21)")
    ensemble.add_argument("--n-theta", type=int, help="initial theta0 grid size (default 20; 1 for fft)")
    ensemble.add_argument("--n-phi", type=int, help="initial phi0 grid size (default 20; 1 for fft)")
    ensemble.add_argument("--burn-frac", type=float, default=0.2, help="fraction of the run discarded as burn-in")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    traj = sub.add_parser("trajectory", parents=[common], help="integrate one trajectory and plot it")
    traj.add_argument("--plot", choices=[*TRAJECTORY_PLOTS, *PHASE_PLOTS], default="u")
    traj.add_argument("--window", type=float, nargs=2, metavar=("T_START", "T_END"), help="tau window to show")

    sub.add_parser("histogram", parents=[common, ensemble], help="ensemble velocity histogram vs Gaussian")

    fft = sub.add_parser("fft", parents=[common, ensemble], help="ensemble-averaged Fourier spectrum")
    fft.add_argument("--mode", choices=["semilog", "loglog", "linear"], default="semilog")
    fft.add_argument("--variable", default="u", choices=["xi", "u", "Gam", "beta", "th", "ph"])

    spectrum = sub.add_parser("spectrum", parents=[common], help="single-trajectory spectrum with exponential fit")
    spectrum.add_argument("--diagnostic", action="store_true", help="overlay sin(t) to check the FFT pipeline")
    spectrum.add_argument("--linear", action="store_true", help="linear instead of log y-axis")

    sub.add_parser("running", parents=[common, ensemble], help="running time averages of x^2, v^2 and Gamma")
    sub.add_parser("kurtosis", parents=[common, ensemble], help="running kurtosis of the velocity")

    lor = sub.add_parser("lorentzian", parents=[common], help="Lorentzian fit of a pulse in u(tau)")
    lor.add_argument("--window", type=float, default=20.0, help="width of the tau window around the pulse")
    return parser


def run(args):
    p = Params(omega_tau0=args.omega_tau, f0=args.f0, alpha=args.alpha, q=args.q, t1=args.t1, dt=args.dt)
    cmd = args.command

    if cmd in ("trajectory", "spectrum", "lorentzian"):
        traj = integrate(p)

    if cmd == "trajectory":
        if args.plot in PHASE_PLOTS:
            t_start, t_end = args.window or (0.2 * p.t1, p.t1)
            PHASE_PLOTS[args.plot](traj, p, t_start, t_end)
        else:
            TRAJECTORY_PLOTS[args.plot](traj, p, xlim=args.window)
        return

    if cmd == "spectrum":
        start = int(0.2 * len(traj.tau))  # skip burn-in
        fourier_transform(traj, p, start=start, diagnostic=args.diagnostic, log=not args.linear)
        return

    if cmd == "lorentzian":
        plot_pulse_with_lorentzian_overlay(traj.tau, traj.u, p, window=args.window)
        return

    # ensemble commands
    u0_vals = np.linspace(-5, 5, args.n_u0)
    default_grid = 1 if cmd == "fft" else 20
    n_theta = args.n_theta or default_grid
    n_phi = args.n_phi or default_grid
    grid = dict(n_theta=n_theta, n_phi=n_phi, burn_frac=args.burn_frac)

    if cmd == "histogram":
        plot_ensemble_avg_with_ideal_gaussian(p, u0_vals, variable="u", xlabel="u", **grid)
    elif cmd == "fft":
        ensemble_fft(p, u0_vals, variable=args.variable, mode=args.mode, **grid)
    elif cmd == "running":
        _, _, v2bar, _ = ensemble_running_time_averages_x2_v2(p, u0_vals, **grid)
        print("final <v^2> running average:", v2bar[-1])
    elif cmd == "kurtosis":
        _, mean_K, _, _ = ensemble_running_kurtosis_v(p, u0_vals, **grid)
        print("Final (late-time) kurtosis estimate:", mean_K[-1])


def finish(args):
    """Show figures, or save them as PNGs when --save is given."""
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for i, num in enumerate(plt.get_fignums(), start=1):
            path = out / f"{args.command}_{i}.png"
            plt.figure(num).savefig(path, dpi=200)
            print("saved", path)
    else:
        plt.show()


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
    finish(args)
