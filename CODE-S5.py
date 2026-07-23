"""
Identifier: Code S5
Purpose: (1) correctly re-test whether ALL d+1 correct MUBs (not the paper's
         slope-basis formula) satisfy S(M_a|B) = S(M(t)) for d=2, across many r.
         (2) separately test whether the slope-basis formula (eq:slope-basis)
         is degenerate for d=2, i.e. fails to produce 3 distinct MUBs.
Reference: Corollary "Exact steering criterion" is unrelated; this file supports Theorem "Complete-MUB saturation for d=2" and the Remark on the scope of the constructive formula, in the manuscript.
Status: VALID. Test 1 confirmed exact equality (gap_X = gap_Y = 0 to floating-point precision) for the Y basis across r=0.1,0.3,0.5,0.7,0.9, and exact vanishing of the Z-basis conditional entropy in every case. Test 2 confirmed degeneracy of the slope-basis formula at d=2: only 2 distinct vectors were generated from the 4 pairs (a,b), against the 2d=4 required for a complete non-degenerate construction.
"""
import numpy as np

def von_neumann_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return -np.sum(ev * np.log(ev))

def M_matrix(d, r):
    n = np.arange(d)
    diff = n[:, None] - n[None, :]
    return (1.0 / d) * r ** (diff ** 2)

def conditional_entropy_for_basis(M, psi):
    """psi: length-d complex vector, a measurement basis state on A.
    Returns S(sigma_normalized) where sigma_un[n,n'] = M[n,n'] * conj(psi[n]) * psi[n']."""
    outer = np.outer(psi.conj(), psi)
    sigma_un = outer * M
    p = np.trace(sigma_un).real
    sigma = sigma_un / p
    return von_neumann_entropy(sigma), p

# ---- Test 1: correct qubit MUBs (Z, X, Y), test each basis SEPARATELY vs S(M) ----
print("=== Test 1: correct Pauli MUBs for d=2, per-basis comparison ===")
for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
    M = M_matrix(2, r)
    S_M = von_neumann_entropy(M)
    Z = np.eye(2, dtype=complex)
    X = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype=complex)
    Y = (1/np.sqrt(2)) * np.array([[1,1],[1j,-1j]], dtype=complex)
    results = {}
    for name, U in [('Z', Z), ('X', X), ('Y', Y)]:
        vals = []
        for k in range(2):
            S_k, p_k = conditional_entropy_for_basis(M, U[:, k])
            vals.append(S_k)
        results[name] = vals
    print(f"r={r:.2f}  S(M)={S_M:.6f}")
    print(f"  Z-basis conditional entropies: {results['Z']}  (expected: [0,0])")
    print(f"  X-basis conditional entropies: {results['X']}  (expected: [S(M),S(M)])")
    print(f"  Y-basis conditional entropies: {results['Y']}  (expected: [S(M),S(M)])")
    gap_X = max(abs(v - S_M) for v in results['X'])
    gap_Y = max(abs(v - S_M) for v in results['Y'])
    print(f"  gap_X={gap_X:.3e}  gap_Y={gap_Y:.3e}")

# ---- Test 2: does the slope-basis formula from the paper degenerate at d=2? ----
print("\n=== Test 2: slope-basis formula degeneracy check for d=2 ===")
d = 2
omega = np.exp(2j*np.pi/d)
generated = []
for a in range(d):
    for b in range(d):
        n = np.arange(d)
        psi = omega**(a*n**2 + b*n) / np.sqrt(d)
        generated.append(((a,b), np.round(psi, 6)))
for (a,b), psi in generated:
    print(f"  a={a} b={b}  psi={psi}")
distinct = set(tuple(np.round(psi,6)) for (_,psi) in generated)
print(f"Number of (a,b) pairs: {len(generated)}, distinct basis vectors: {len(distinct)}")
print("If distinct < 2*d, the formula is degenerate for d=2 (fails to build the full MUB set).")