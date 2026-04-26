import numpy as np
from fitting_method import newton_raphson
from integrator import integrate_outward, integrate_inward
from plots import plot_structure

# Solar constants (CGS)
M_sun  = 1.989e33
R_sun  = 6.957e10
L_sun  = 3.828e33

# Composition
X, Y, Z = 0.70, 0.28, 0.02

M   = M_sun
M_f = 0.5 * M_sun

# Initial guesses
rho_c0 = 150.0
T_c0   = 1.5e7
R0     = R_sun
L_s0   = L_sun

print("Running Newton-Raphson...")
rho_c, T_c, R, L_s = newton_raphson(rho_c0, T_c0, R0, L_s0, M, M_f, X, Y, Z)

print()
print(f"{'':20s}  {'Model':>12s}  {'Solar':>12s}  {'Ratio':>8s}")
print("-" * 58)
print(f"{'rho_c (g/cm^3)':20s}  {rho_c:12.4e}  {'~150':>12s}")
print(f"{'T_c (K)':20s}  {T_c:12.4e}  {'~1.57e7':>12s}")
print(f"{'R (cm)':20s}  {R:12.4e}  {R_sun:12.4e}  {R/R_sun:8.4f}")
print(f"{'L (erg/s)':20s}  {L_s:12.4e}  {L_sun:12.4e}  {L_s/L_sun:8.4f}")

# Integrate final model for plotting
M_c    = 1e-10 * M
sol_in  = integrate_outward(M_c, M_f, rho_c, T_c, X, Y, Z)
sol_out = integrate_inward(M, M_f, R, L_s, X, Y, Z)

plot_structure(sol_in, sol_out, M, R, L_s, M_f)
