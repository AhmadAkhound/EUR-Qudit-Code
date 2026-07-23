"""
Identifier: Code S4
Purpose: Broad robustness sweep of the exact EUR saturation identity
    S(M2|B) = S(M(t)) across 420 parameter combinations: six
    environmental parameter regimes, five independent time points, and
    dimensions d=2 to d=15.
Used in: Section "Exact saturation of Berta's bound", broad numerical
    verification subsection.
Status: Valid and used in the manuscript. Maximum |Gap| across all 420
    checks: 2.6e-12 (numerical noise floor, not a physical deviation).
"""
import numpy as np
from scipy import integrate

def Gamma(t, omega0, lam, eta):
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-6, 50*omega0, limit=400)
    return val

def r_of_t(t, omega0, lam, eta):
    return np.exp(-2*Gamma(t, omega0, lam, eta))

def entropy_toeplitz(r, d):
    k = np.arange(d)
    M = r ** ((k[:, None] - k[None, :]) ** 2) / d
    eigs = np.linalg.eigvalsh(M)
    eigs = np.real(eigs)
    eigs = eigs[eigs > 1e-14]
    return -np.sum(eigs * np.log(eigs))

def S_M2_given_B(r, d):
    """S(M2|B) via the verified compact formula (closed form, no optimization)."""
    omega = np.exp(2j * np.pi / d)
    fourier_basis = [np.array([omega**(n*k) for n in range(d)]) / np.sqrt(d) for k in range(d)]
    rho_XB = np.zeros((d, d, d), dtype=complex)
    for i, psi in enumerate(fourier_basis):
        for b in range(d):
            for bp in range(d):
                rho_XB[i, b, bp] = np.conj(psi[b]) * psi[bp] * (1/d) * r**((b-bp)**2)
    p = np.array([np.real(np.trace(rho_XB[i])) for i in range(d)])
    S_XB = 0.0
    for i in range(d):
        pi = p[i]
        if pi > 1e-13:
            eigs = np.linalg.eigvalsh(rho_XB[i] / pi)
            eigs = eigs[eigs > 1e-13]
            S_XB += pi * (-np.log(pi) - np.sum(eigs * np.log(eigs)))
    rho_B = np.sum(rho_XB, axis=0)
    eigs_B = np.linalg.eigvalsh(rho_B)
    eigs_B = np.real(eigs_B)
    eigs_B = eigs_B[eigs_B > 1e-14]
    S_B = -np.sum(eigs_B * np.log(eigs_B))
    return S_XB - S_B

if __name__ == "__main__":
    param_sets = [
        dict(omega0=1.0, lam=0.1, eta=0.15),
        dict(omega0=1.0, lam=0.1, eta=0.05),
        dict(omega0=1.0, lam=0.1, eta=0.35),
        dict(omega0=1.0, lam=0.05, eta=0.15),
        dict(omega0=1.0, lam=0.25, eta=0.15),
        dict(omega0=2.0, lam=0.2, eta=0.15),
    ]
    dims = list(range(2, 16))          # d = 2..15
    times = [0.5, 1.5, 3.0, 6.0, 10.0]  # 5 independent time points

    total_checks = 0
    max_gap = 0.0
    worst_case = None

    for pset in param_sets:
        for t in times:
            r = r_of_t(t, **pset)
            for d in dims:
                S_toep = entropy_toeplitz(r, d)
                S_M2B = S_M2_given_B(r, d)
                gap = abs(S_M2B - S_toep)
                total_checks += 1
                if gap > max_gap:
                    max_gap = gap
                    worst_case = (pset, t, d, gap)

    print(f"Total checks performed: {total_checks}")
    print(f"Maximum |Gap| across all cases: {max_gap:.3e}")
    print(f"Worst case: parameters={worst_case[0]}  t={worst_case[1]}  d={worst_case[2]}  gap={worst_case[3]:.3e}")
    print(f"\n>>> Result: {'Equality confirmed in all '+str(total_checks)+' cases, "Equality confirmed within numerical precision"' if max_gap < 1e-8 else 'A violation was found - further investigation needed!'}")