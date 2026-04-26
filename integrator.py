import numpy as np
from scipy.integrate import solve_ivp
from microphysics import (G, a_rad, pi, sigma_SB,
                           calculate_eos, get_rho_from_PT,
                           calculate_opacity, calculate_nuclear_rates,
                           calculate_nabla)


def stellar_odes(M_r, state, X, Y, Z):
    """
    RHS of the four stellar structure equations.
    Independent variable: interior mass M_r (g)
    State: [r (cm), P (dyn/cm^2), L (erg/s), T (K)]
    """
    r, P, L, T = state

    if r <= 0 or P <= 0 or T <= 0 or M_r <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    rho = get_rho_from_PT(P, T, X, Y, Z)
    kappa = calculate_opacity(rho, T, X, Y, Z)
    epsilon = calculate_nuclear_rates(rho, T, X, Y, Z)
    nabla = calculate_nabla(rho, T, P, kappa, L, M_r)

    dr = 1.0 / (4.0 * pi * r**2 * rho)
    dP = -G * M_r / (4.0 * pi * r**4)
    dL = epsilon
    dT = -(G * M_r * T * nabla) / (4.0 * pi * r**4 * P)

    return [dr, dP, dL, dT]


def central_bcs(M_c, rho_c, T_c, X, Y, Z):
    """
    Central boundary conditions from power-series expansion at small M_c.

    Parameterized by (rho_c, T_c). Returns [r, P, L, T] at M_c.

    Expansions (leading order in M_c):
        r ~ r1 * M_c^(1/3)
        P ~ P_c - (3G / 8pi r1^4) * M_c^(2/3)
        L ~ epsilon_c * M_c
        T ~ T_c - (3G T_c nabla_c / 8pi r1^4 P_c) * M_c^(2/3)
    """
    P_c = calculate_eos(rho_c, T_c, X, Y, Z)
    kappa_c = calculate_opacity(rho_c, T_c, X, Y, Z)
    epsilon_c = calculate_nuclear_rates(rho_c, T_c, X, Y, Z)

    # nabla_c: L = epsilon_c * M_r cancels M_r in nabla_rad, giving a finite limit
    nabla_c = calculate_nabla(rho_c, T_c, P_c, kappa_c,
                               L=epsilon_c * M_c, M_r=M_c)

    r1 = (3.0 / (4.0 * pi * rho_c))**(1.0 / 3.0)
    coeff = 3.0 * G / (8.0 * pi * r1**4)   # shared coefficient for P and T expansions

    r_c = r1 * M_c**(1.0 / 3.0)
    P_c_start = P_c - coeff * M_c**(2.0 / 3.0)
    L_c_start = epsilon_c * M_c
    T_c_start = T_c - coeff * (T_c * nabla_c / P_c) * M_c**(2.0 / 3.0)

    return [r_c, P_c_start, L_c_start, T_c_start]


def surface_bcs(M, R, L_s, X, Y, Z):
    """
    Surface boundary conditions at M_r = M (photosphere).

    Parameterized by (R, L_s). Returns [r, P, L, T] at M.

    At photosphere (tau = 2/3), Eddington approximation:
        T_eff = (L / 4*pi*R^2*sigma)^(1/4)
        P_ph  = (2/3) * g / kappa_ph,  g = GM/R^2
    kappa_ph solved iteratively (~5 iters to converge).
    """
    g = G * M / R**2
    T_eff = (L_s / (4.0 * pi * R**2 * sigma_SB))**0.25
    kappa_es = 0.2 * (1.0 + X)
    P_ph = (2.0 / 3.0) * g / kappa_es

    return [R, P_ph, L_s, T_eff]


def integrate_outward(M_c, M_f, rho_c, T_c, X, Y, Z):
    """Integrate outward from M_c to M_f using central BCs."""
    state0 = central_bcs(M_c, rho_c, T_c, X, Y, Z)

    def hit_zero(M_r, y, *args): return min(y)
    hit_zero.terminal  = True
    hit_zero.direction = -1

    return solve_ivp(stellar_odes, (M_c, M_f), state0,
                     args=(X, Y, Z), method='RK45',
                     events=hit_zero, rtol=1e-6, atol=1e-10)


def integrate_inward(M, M_f, R, L_s, X, Y, Z):
    """Integrate inward from M to M_f using surface BCs."""
    state0 = surface_bcs(M, R, L_s, X, Y, Z)

    def hit_zero(M_r, y, *args): return min(y)
    hit_zero.terminal  = True
    hit_zero.direction = -1

    return solve_ivp(stellar_odes, (M, M_f), state0,
                     args=(X, Y, Z), method='RK45',
                     events=hit_zero, rtol=1e-6, atol=1e-10)
