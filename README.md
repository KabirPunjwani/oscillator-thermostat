# Chaotic Thermostat Dynamics: Van der Pol Oscillator and Martyna–Klein–Tuckerman (MKT) chaotic thermostat Thermostat

Computational plasma physics research (UCLA, advised by Prof. George Morales) on how a deterministic, low-dimensional system can reproduce statistical behavior (Gaussian/Maxwellian velocity distributions) and how to *quantify* when it is chaotic versus periodic.

## The question

A Van der Pol-type oscillator is coupled to a chaotic thermostat (the MKT system) that supplies a fluctuating friction `Γ` and two phase variables `θ`, `φ`. The full system is six coupled ODEs in dimensionless form, with three main control parameters: the thermostat timescale `ω_τ` and parameters `q` and `⍺`.

I wanted to learn:
1. Does the oscillator's velocity develop a Maxwellian (Gaussian) distribution despite the dynamics being fully deterministic?
2. Can I tell chaotic from periodic regimes using data-driven diagnostics alone, and does that agree with an analytic prediction? 

## Approach and why

- **Simulation:** hand-written RK4 integrator in NumPy, vectorized to run hundreds of trajectories in parallel (one batch state array, no per-trajectory Python loop).
- **Ensembles:** chaotic systems are sensitive to initial conditions, so single trajectories are not trustworthy statistics. I average over a 20x20 grid of initial phases (`θ0`, `φ0`) across 21 initial velocities, with burn-in removed.
- **Several independent diagnostics**, because no single chaos test is reliable on its own:
  - Ensemble-averaged velocity histograms vs. an ideal Gaussian
  - Running time-averages of `x²`, `v²`, and `Γ` (ergodicity check) and running kurtosis of `v`
  - Hann-windowed FFT spectra with log-linear fits of the spectral tail to extract a chaos timescale `τ_L`, cross-checked against a Lorentzian pulse-width fit in the time domain

## Validation

- Tested the FFT pipeline on a known `sin(t)` signal to confirm peak location and amplitude normalization (`python main.py spectrum --diagnostic`).
- Cross-checked the spectral timescale `τ_L` against an independent time-domain Lorentzian pulse-width estimate.
- Confirmed results are stable across initial conditions via the ensemble rather than relying on one run.

## Results

For a detailed breakdown of the simulation outcomes, please view the [Findings Summary](./VdP_MKT_Findings.pdf).

## Project structure

```
main.py                    command-line entry point (one subcommand per analysis)
thermostat/
  params.py                model parameters (Params dataclass), state-vector layout
  system.py                the six coupled ODEs (single trajectory + vectorized batch)
  integrators.py           RK4 stepper
  simulate.py              integrate one trajectory; run phase ensembles in parallel
  plotting.py              time series and phase portraits
  distributions.py         ensemble histograms vs. ideal Gaussian
  spectral.py              Hann-windowed FFT, exponential / power-law fits, ensemble spectra
  lorentzian.py            time-domain Lorentzian pulse-width estimate of tau_L
  time_averages.py         running time averages and running kurtosis
```

## Running it

```bash
pip install -r requirements.txt

python main.py trajectory --plot u            # single trajectory (also: xi, gam, phase, ...)
python main.py histogram --save figures/      # ensemble velocity histogram vs Gaussian
python main.py fft --mode loglog              # ensemble-averaged spectrum + fits
python main.py spectrum --diagnostic          # single-trajectory spectrum, FFT sanity check
python main.py running                        # running time averages of x^2, v^2, Gamma
python main.py kurtosis                       # running kurtosis of the velocity
python main.py lorentzian                     # Lorentzian pulse-width fit
```

Parameters are flags (`--omega-tau`, `--q`, `--alpha`, `--f0`, `--t1`, `--dt`); `--save DIR` writes PNGs instead of opening windows. Full ensemble runs take a few minutes, so use a shorter run (e.g. `--t1 400`) for a quick look. Everything is also importable:

```python
from thermostat import Params, integrate, ensemble_running_kurtosis_v
import numpy as np

p = Params(omega_tau0=10, q=0.5)
t, K, _, _ = ensemble_running_kurtosis_v(p, np.linspace(-5, 5, 21), show_plot=False)
```

## Tech

Python, NumPy, Matplotlib. Numerical integration (RK4), vectorized simulation, spectral analysis, time-series statistics.
