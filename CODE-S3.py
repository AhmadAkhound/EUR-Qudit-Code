"""
Identifier: Code S3 -- Figure generation
Purpose: Generates Figure 4 (Berta bound saturation: S(M1|B)+S(M2|B) vs S(M(t)))
    and Figure 5 (complementary relation: D(t) and [S(M1|B)+S(M2|B)] summing
    to ln d).
Used in: Figures 4 and 5.
Status: VALID. Figure 4 independently compares the Fourier-branch
    conditional entropy with S(M(t)); the observed gap is of order 1e-15.
    Figure 5 illustrates the analytically derived complementary identity and
    s not presented as an independent numerical verification of that identity.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def Gamma(t, omega0=1.0, lam=0.1, eta=0.15):
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-6, 50*omega0, limit=400, epsabs=1e-14, epsrel=1e-14)
    return val

def r_of_t(t, **kw):
    return np.exp(-2*Gamma(t, **kw))

def entropy_toeplitz(r, d):
    k = np.arange(d)
    M = r**((k[:,None]-k[None,:])**2) / d
    eigs = np.linalg.eigvalsh(M)
    eigs = eigs[eigs>1e-14]
    return -np.sum(eigs*np.log(eigs))

def discord_closed_form(r, d):
    return np.log(d) - entropy_toeplitz(r, d)

# ---------- Settings ----------
d = 4
ts = np.linspace(0.01, 15, 500)
rs = np.array([r_of_t(t) for t in ts])

def S_M2_given_B(r, d):
    """Independent computation of S(M2|B) via the Fourier-branch formula (not copied from S(M(t)))."""
    omega = np.exp(2j*np.pi/d)
    fourier_basis = [np.array([omega**(n*k) for n in range(d)])/np.sqrt(d) for k in range(d)]
    rho_XB = np.zeros((d,d,d), dtype=complex)
    for i, psi in enumerate(fourier_basis):
        for b in range(d):
            for bp in range(d):
                rho_XB[i,b,bp] = np.conj(psi[b])*psi[bp]*(1/d)*r**((b-bp)**2)
    p = np.array([np.real(np.trace(rho_XB[i])) for i in range(d)])
    S_XB = 0.0
    for i in range(d):
        pi = p[i]
        if pi > 1e-13:
            eigs = np.linalg.eigvalsh(rho_XB[i]/pi)
            eigs = eigs[eigs>1e-13]
            S_XB += pi*(-np.log(pi) - np.sum(eigs*np.log(eigs)))
    rho_B = np.sum(rho_XB, axis=0)
    eigs_B = np.linalg.eigvalsh(rho_B)
    eigs_B = eigs_B[eigs_B>1e-14]
    S_B = -np.sum(eigs_B*np.log(eigs_B))
    return S_XB - S_B

S_M = np.array([entropy_toeplitz(r, d) for r in rs])
S_M1B = np.zeros_like(S_M)                          # proven exactly zero (Proposition SM1B-zero)
S_M2B = np.array([S_M2_given_B(r, d) for r in rs])   # independently computed, NOT copied from S_M
D_t = np.log(d) - S_M                                 # closed-form discord

# ---------- Figure 4: saturation of Berta's bound ----------
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(ts, S_M1B + S_M2B, label=r'$S(M_1|B)+S(M_2|B)$', lw=2.5)
ax.plot(ts, S_M, '--', label=r'$S(M(t))$', lw=1.5)
ax.set_xlabel(r'$t$')
ax.set_ylabel('Entropy (nats)')
ax.set_title(f'Exact saturation of Berta bound (d={d})')
ax.legend()
plt.tight_layout()
plt.savefig('figure4_berta_saturation.pdf', dpi=300)
plt.savefig('figure4_berta_saturation.png', dpi=300)
plt.show()

# ---------- Figure 5: complementary relation ----------
fig, ax = plt.subplots(figsize=(7,4.5))
ax.plot(ts, D_t, label=r'$D(t)$ (discord)', lw=2)
ax.plot(ts, S_M1B + S_M2B, label=r'$S(M_1|B)+S(M_2|B)$ (uncertainty)', lw=2)
ax.axhline(np.log(d), color='k', ls=':', label=r'$\ln d$')
ax.plot(ts, D_t + S_M1B + S_M2B, 'r--', lw=1, label='sum (should be flat)')
ax.set_xlabel(r'$t$')
ax.set_ylabel('Value (nats)')
ax.set_title(f'Complementary relation: D(t)+[S(M1|B)+S(M2|B)]=ln(d)  (d={d})')
ax.legend()
plt.tight_layout()
plt.savefig('figure5_complementary.pdf', dpi=300)
plt.savefig('figure5_complementary.png', dpi=300)
plt.show()
from scipy.signal import argrelextrema

print("=== Figure 4: extrema of S(M1|B)+S(M2|B) and S(M(t)) ===")
for name, arr in [("Sum(M1+M2|B)", S_M1B+S_M2B), ("S(M(t))", S_M)]:
    maxi = argrelextrema(arr, np.greater)[0][:3]
    mini = argrelextrema(arr, np.less)[0][:3]
    print(f"{name}: max=", [(round(ts[i],2), round(arr[i],6)) for i in maxi],
          " min=", [(round(ts[i],2), round(arr[i],6)) for i in mini])
print("Maximum |Gap| over the full range:", np.max(np.abs((S_M1B+S_M2B) - S_M)))

print("\n=== Figure 5: extrema of D(t) and the sum (should be constant = ln(d)) ===")
for name, arr in [("D(t)", D_t), ("D+Sum", D_t + S_M1B + S_M2B)]:
    maxi = argrelextrema(arr, np.greater)[0][:3]
    mini = argrelextrema(arr, np.less)[0][:3]
    print(f"{name}: max=", [(round(ts[i],2), round(arr[i],6)) for i in maxi],
          " min=", [(round(ts[i],2), round(arr[i],6)) for i in mini])
print(f"ln(d)={np.log(d):.6f}   Maximum deviation of the sum from ln(d):", np.max(np.abs(D_t+S_M1B+S_M2B - np.log(d))))
print("Figures saved: figure4_berta_saturation.{pdf,png}, figure5_complementary.{pdf,png}")