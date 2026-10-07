"""Model parameters and state-vector conventions."""

from dataclasses import dataclass

import numpy as np

# Column order of the 6-component state vector z = [xi, u, Gam, beta, th, ph]
STATE_COLUMNS = {"xi": 0, "u": 1, "Gam": 2, "beta": 3, "th": 4, "ph": 5}


def column(variable: str) -> int:
    """Return the state-vector column index for a variable name."""
    try:
        return STATE_COLUMNS[variable]
    except KeyError:
        raise ValueError(f"variable must be one of {list(STATE_COLUMNS)}") from None


@dataclass(frozen=True)
class Params:
    """Dimensionless parameters of the coupled VdP + MKT thermostat system."""

    omega_tau0: float = 10.0  # thermostat timescale (omega * tau)
    f0: float = 0.2           # phase-drive amplitude
    alpha: float = 7.0        # Van der Pol strength
    q: float = 0.5            # damping / nonlinearity parameter q
    t0: float = 0.0           # integration start (tau)
    t1: float = 1200.0        # integration end (tau)
    dt: float = 0.01          # RK4 step

    @property
    def n_steps(self) -> int:
        """Number of stored time points, including t0."""
        return int(np.ceil((self.t1 - self.t0) / self.dt)) + 1

    @property
    def label(self) -> str:
        """Short parameter string used in plot titles."""
        return f"omegatau = {self.omega_tau0:g} and q {self.q:g}"
