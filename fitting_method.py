import numpy as np
from integrator import integrate_outward, integrate_inward


def mismatch(rho_c, T_c, R, L_s, M, M_f, X, Y, Z):
    """
    Four residuals at fitting point M_f.
    F_i = ln(q_in_i / q_out_i) for [r, P, L, T].
    """
    M_c = 1e-10 * M

    sol_in  = integrate_outward(M_c, M_f, rho_c, T_c, X, Y, Z)
    sol_out = integrate_inward(M, M_f, R, L_s, X, Y, Z)

    if not sol_in.success or not sol_out.success:
        return np.full(4, 1e10)

    # Check both integrations actually reached M_f (not terminated early by event)
    if abs(sol_in.t[-1] - M_f) > 1e-3 * M_f or abs(sol_out.t[-1] - M_f) > 1e-3 * M_f:
        return np.full(4, 1e10)

    q_in  = sol_in.y[:, -1]
    q_out = sol_out.y[:, -1]

    if np.any(q_in <= 0) or np.any(q_out <= 0):
        return np.full(4, 1e10)

    F1 = np.log(q_in[0] / q_out[0])  # r
    F2 = np.log(q_in[1] / q_out[1])  # P
    F3 = np.log(q_in[2] / q_out[2])  # L
    F4 = np.log(q_in[3] / q_out[3])  # T

    return np.array([F1, F2, F3, F4])


def _jacobian(F_func, p, F0, eps=1e-4):
    """Numerical Jacobian via forward differences."""
    J = np.zeros((len(p), len(p)))
    for j in range(len(p)):
        p_pert = p.copy()
        p_pert[j] += eps
        J[:, j] = (F_func(p_pert) - F0) / eps
    return J


def newton_raphson(rho_c0, T_c0, R0, L_s0, M, M_f, X, Y, Z,
                   tol=1e-6, max_iter=50):
    """
    Newton-Raphson in log-space to converge the four mismatch residuals.
    Returns (rho_c, T_c, R, L_s) at convergence.
    """
    p = np.log([rho_c0, T_c0, R0, L_s0])

    def F_func(p):
        rho_c, T_c, R, L_s = np.exp(p)
        return mismatch(rho_c, T_c, R, L_s, M, M_f, X, Y, Z)

    for i in range(max_iter):
        F = F_func(p)
        print(f"  iter {i:2d}: max|F|={np.max(np.abs(F)):.3e}  F={np.round(F,3)}")
        if np.max(np.abs(F)) < tol:
            print(f"Converged in {i} iterations, max|F| = {np.max(np.abs(F)):.2e}")
            return tuple(np.exp(p))

        J  = _jacobian(F_func, p, F)
        dp = np.linalg.solve(J, -F)
        dp = dp * min(1.0, 0.5 / np.max(np.abs(dp)))  # scale to preserve direction

        # Backtracking line search
        alpha   = 1.0
        F0_norm = np.dot(F, F)
        for _ in range(20):
            F_trial = F_func(p + alpha * dp)
            if np.dot(F_trial, F_trial) < F0_norm:
                break
            alpha *= 0.5

        p = p + alpha * dp

    print(f"Warning: did not converge after {max_iter} iterations")
    return tuple(np.exp(p))
