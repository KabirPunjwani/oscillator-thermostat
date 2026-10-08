"""Van der Pol oscillator coupled to an MKT chaotic thermostat: simulation and analysis."""

from .distributions import (
    ensemble_samples,
    plot_data_with_ideal_maxwellian,
    plot_ensemble_avg_with_ideal_gaussian,
)
from .integrators import rk4_step
from .lorentzian import lorentzian_width_from_timeseries, plot_pulse_with_lorentzian_overlay
from .params import STATE_COLUMNS, Params
from .plotting import (
    gam_plot,
    gam_u_series_plot,
    phase_portrait,
    phase_portrait_gam_u,
    q_minus_gamma_plot,
    u_plot,
    xi_plot,
)
from .simulate import DEFAULT_Z0, Trajectory, evolve_batch, initial_batch, integrate, phase_grid
from .spectral import ensemble_fft, ensemble_spectrum, fit_exponential, fit_power_law, fourier_transform, one_sided_spectrum
from .system import rhs, rhs_batch
from .tarnopolski import (
    abbe_value,
    plot_T_vs_A,
    simulate_pure_vdp,
    tarnopolski_path,
    turning_point_fraction,
)
from .time_averages import ensemble_running_kurtosis_v, ensemble_running_time_averages_x2_v2

__all__ = [
    "abbe_value",
    "plot_T_vs_A",
    "simulate_pure_vdp",
    "tarnopolski_path",
    "turning_point_fraction",
    "DEFAULT_Z0",
    "STATE_COLUMNS",
    "Params",
    "Trajectory",
    "ensemble_fft",
    "ensemble_running_kurtosis_v",
    "ensemble_running_time_averages_x2_v2",
    "ensemble_samples",
    "ensemble_spectrum",
    "evolve_batch",
    "fit_exponential",
    "fit_power_law",
    "fourier_transform",
    "gam_plot",
    "gam_u_series_plot",
    "initial_batch",
    "integrate",
    "lorentzian_width_from_timeseries",
    "one_sided_spectrum",
    "phase_grid",
    "phase_portrait",
    "phase_portrait_gam_u",
    "plot_data_with_ideal_maxwellian",
    "plot_ensemble_avg_with_ideal_gaussian",
    "plot_pulse_with_lorentzian_overlay",
    "q_minus_gamma_plot",
    "rhs",
    "rhs_batch",
    "rk4_step",
    "u_plot",
    "xi_plot",
]
