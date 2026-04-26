import numpy as np

# Physical constants (CGS)
G = 6.674e-8
c_light = 2.998e10
k_B = 1.381e-16
m_H = 1.673e-24
a_rad = 7.566e-15
sigma_SB = 5.671e-5
pi = np.pi



def calculate_eos(rho, T, X, Y, Z):
    """Ideal gas + radiation EOS. Returns P."""
    mu = 1.0 / (2.0*X + 0.75*Y + 0.5*Z)
    P_gas = (rho * k_B * T) / (mu * m_H)
    P_rad = (a_rad * T**4) / 3.0
    P = P_gas + P_rad
    beta = P_gas / P
    return P


def get_rho_from_PT(P, T, X, Y, Z):
    """Invert EOS: given P and T, return density."""
    mu = 1.0 / (2.0*X + 0.75*Y + 0.5*Z)
    P_rad = a_rad * T**4 / 3.0
    P_gas = max(P - P_rad, 1e-30)
    rho = P_gas * mu * m_H / (k_B * T)
    return max(rho, 1e-30)


def calculate_opacity(rho, T, X, Y, Z):
    """Kramers' (ff + bf), electron scattering, and H- opacity."""
    kappa_es = 0.2 * (1.0 + X)
    if T < 1e4:
        # H- dominates in cool photospheric layers (metals supply electrons)
        kappa_Hminus = 2.5e-31 * (Z / 0.02) * rho**0.5 * T**9
        return max(kappa_es + kappa_Hminus, 1e-10)
    else:
        # Free-free Kramers valid from ~10^4 K through the stellar interior
        kappa_ff = 3.68e22 * (1.0 - Z) * (1.0 + X) * rho * T**(-3.5)
        return max(kappa_es + kappa_ff, 1e-10)


def calculate_nuclear_rates(rho, T, X, Y, Z):
    """PP + CNO nuclear energy generation rate (erg/g/s)."""
    XCNO = 0.74 * Z
    rate0_pp = 1.05e-5
    rate0_CNO = 8.24e-24
    T6 = T / 1e6
    if T6 <= 0:
        return 0.0
    rate_pp = rate0_pp* rho *X**2 *T6**4
    rate_cno = rate0_CNO * rho * X * XCNO * T6**16
    return min(rate_pp + rate_cno, 1e5)


def calculate_nabla(rho, T, P, kappa, L, M_r):
    """Actual temperature gradient (min of radiative and adiabatic)."""
    nabla_ad = 0.4
    nabla_rad = (3.0 * kappa * L * P) / (16.0 * pi * a_rad * c_light * G * M_r * T**4)
    return min(nabla_rad, nabla_ad)
