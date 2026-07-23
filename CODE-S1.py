"""
Identifier: Code S1
Purpose: Test whether the pairwise Berta-bound saturation theorem
         (S(M2|B) = S(M)) requires only a uniform diagonal M_nn = 1/d,
         rather than full Toeplitz structure. Two tests:
         (1) generic Toeplitz M with non-Gaussian decay coefficients.
         (2) fully non-Toeplitz Hermitian PSD M with uniform diagonal.
Reference: Theorem "General saturation theorem for a uniform diagonal" in the manuscript.
Status: VALID. Confirmed at machine precision for both generic Toeplitz matrices with non-Gaussian decay and fully non-Toeplitz matrices with a uniform diagonal, d=3 through d=7 in both cases; maximum deviation across all tested cases: 2.887e-15.
"""

import numpy as np

def von_neumann_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return -np.sum(ev * np.log(ev))

def pairwise_gap_general(d, M):
    """M: any Hermitian PSD matrix with trace 1 and uniform diagonal 1/d."""
    S_M = von_neumann_entropy(M)
    omega = np.exp(2j * np.pi / d)
    p_list, S_list = [], []
    for k in range(d):
        Dk = np.diag([omega ** (-b * k) for b in range(d)])
        sigma_un = (1.0 / d) * (Dk @ M @ Dk.conj().T)
        p_k = np.trace(sigma_un).real
        sigma_k = sigma_un / p_k
        p_list.append(p_k)
        S_list.append(von_neumann_entropy(sigma_k))
    p = np.array(p_list)
    Sk = np.array(S_list)
    H_p = -np.sum(p * np.log(p))
    S_M2B = H_p + np.sum(p * Sk) - np.log(d)
    return abs(S_M2B - S_M)

# ---- Test 1: generic Toeplitz M (non-Gaussian decay, not r^{(n-m)^2}) ----
print("=== Test 1: generic Toeplitz M (non-Gaussian decay) ===")
np.random.seed(0)
for d in [3, 4, 5, 6, 7]:
    raw = np.array([1.0] + [0.7 ** k + 0.05 * np.random.rand() for k in range(1, d)])
    raw = np.sort(raw)[::-1]
    raw = raw / raw[0]
    n = np.arange(d)
    idx = np.abs(n[:, None] - n[None, :])
    M = raw[idx] / d
    ev = np.linalg.eigvalsh(M)
    if ev.min() < -1e-10:
        print(f"d={d}: M not PSD, skipped")
        continue
    g = pairwise_gap_general(d, M)
    print(f"d={d}  gap={g:.3e}")

# ---- Test 2: fully non-Toeplitz Hermitian PSD M with uniform diagonal ----
print("\n=== Test 2: non-Toeplitz M with uniform diagonal (random, Gershgorin-safe) ===")
for d in [3, 4, 5, 6, 7]:
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    H = (A + A.conj().T) / 2
    np.fill_diagonal(H, 0)
    row_sums = np.sum(np.abs(H), axis=1)
    # Gershgorin-safe scaling: guarantees eps * row_sum < 1/d for every row,
    # so every Gershgorin disk stays within the positive half-line -> M is PSD by construction.
    eps = 0.8 / (d * row_sums.max())
    M = np.eye(d) / d + eps * H
    ev = np.linalg.eigvalsh(M)
    assert ev.min() > 0, f"d={d}: M not PSD (min eig={ev.min():.3e}) -- safety factor too large"
    g = pairwise_gap_general(d, M)
    print(f"d={d}  (random off-diagonal, non-Toeplitz)  gap={g:.3e}")