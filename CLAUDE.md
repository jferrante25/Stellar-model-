 Stellar Structure Model
## Project description
This project computes ZAMS stellar models using the fitting
method. The code integrates the four stellar structure equations
(mass, hydrostatic equilibrium, energy generation, and heat
transport) from both the center and the surface, matching at
an interior fitting point using Newton-Raphson iteration.
## Language and dependencies
- Python 3, using numpy, scipy, matplotlib
- No external stellar-physics packages; all microphysics is
implemented from scratch
## Code structure
- microphysics.py: equation of state (ideal gas + radiation),
Kramers’ + electron scattering + H- opacity, pp + CNO energy
generation
- integrator.py: ODE integration using scipy.integrate.solve_ivp
- fitting_method.py: central expansions, surface BCs, mismatch
residuals, Jacobian, Newton-Raphson in log-spac
- run_solar_model.py: main driver that calls the fitting method
- plots.py: structure profile plots
## How to run
python run_solar_model.py
## Physical conventions
- CGS units throughout
- Independent variable: interior mass M_r
- Composition: X = 0.70, Y = 0.28, Z = 0.02 (solar)
- Fitting point at M_f = 0.5 * M
- Convergence criterion: max|F_i| < 1e-6
## Key numerical techniques
- Log-ratio residuals: F_i = ln(q_in / q_out)
- Newton-Raphson iteration in log-space for positivity
- Backtracking line search with step capping
- Energy generation rate capped at 1e5 erg/g/s
- Termination events to stop integration if variables go
negative

