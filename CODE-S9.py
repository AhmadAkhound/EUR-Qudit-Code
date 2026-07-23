"""
Identifier: Code S9
Purpose: Numerically verifies the strict entropy-deficit relation
S(M(t)) < ln(d) for r(t)>0, and hence the positivity of
D(t)=ln(d)-S(M(t)), across several dimensions and time points.
This code does not test an LHS model or establish an EPR-steering criterion.
"""
import numpy as np
from scipy import integrate

def Gamma(t, omega0=1.0, lam=0.1, eta=0.15):
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-6, 50*omega0, limit=400)
    return val

def r_of_t(t, **kw):
    return np.exp(-2*Gamma(t, **kw))

def entropy_toeplitz(r, d):
    k = np.arange(d)
    M = r**((k[:,None]-k[None,:])**2) / d
    eigs = np.linalg.eigvalsh(M)
    eigs = eigs[eigs>1e-14]
    return -np.sum(eigs*np.log(eigs))

if __name__ == "__main__":
    for d in [2, 4, 8]:
        print(f'--- d={d} (ln d = {np.log(d):.4f}) ---')
        for t in [0.5, 3.32, 6.28, 9.68, 15]:
            r = r_of_t(t)
            S_M = entropy_toeplitz(r, d)
            entropy_deficit = np.log(d) - S_M
            strict_deficit = entropy_deficit > 0
            print(
            f'  t={t:5.2f}  r={r:.4f}  S(M(t))={S_M:.5f}  '
             f'ln(d)-S(M)={entropy_deficit:.5f}  Positive={strict_deficit}'
            )