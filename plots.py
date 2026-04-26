import numpy as np
import matplotlib.pyplot as plt

# Known solar central values for comparison
RHO_C_SSM = 150.0
T_C_SSM   = 1.57e7
P_C_SSM   = 2.5e17


def plot_structure(sol_in, sol_out, M, R, L_s, M_f):
    m_in  = sol_in.t  / M
    r_in, P_in, L_in, T_in = sol_in.y

    m_out = sol_out.t / M
    r_out, P_out, L_out, T_out = sol_out.y

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Solar Structure Model")

    labels_in  = ("Outward (center)", "Inward (surface)")
    labels_out = ("Inward (surface)",)

    def panel(ax, m1, q1, m2, q2, ylabel, solar_val=None):
        ax.plot(m1, q1, label="Outward")
        ax.plot(m2, q2, label="Inward", linestyle="--")
        if solar_val is not None:
            ax.axhline(solar_val, color="k", linestyle=":", linewidth=1,
                       label=f"SSM: {solar_val:.2e}")
        ax.axvline(m_in[-1] if len(m_in) else 0.5, color="gray",
                   linestyle=":", linewidth=0.8, label=f"M_f = {M_f/M:.1f} M")
        ax.set_xlabel(r"$M_r / M_\odot$")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    panel(axes[0, 0], m_in, r_in,  m_out, r_out, r"$r$ (cm)",       solar_val=R)
    panel(axes[0, 1], m_in, P_in,  m_out, P_out, r"$P$ (dyn/cm²)",  solar_val=P_C_SSM)
    panel(axes[1, 0], m_in, L_in,  m_out, L_out, r"$L$ (erg/s)",    solar_val=L_s)
    panel(axes[1, 1], m_in, T_in,  m_out, T_out, r"$T$ (K)",        solar_val=T_C_SSM)

    plt.tight_layout()
    plt.savefig("stellar_structure.png", dpi=150)
    print("Saved stellar_structure.png")
    plt.show()
