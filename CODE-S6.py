"""
Identifier: Code S6
Purpose: Preliminary test using several maximal commuting subgroups of the
         2-qubit Pauli group for d=4, as candidate bases beyond the
         computational one, checked against thm:general-diagonal-saturation.
         Not all such subgroups yield bases mutually unbiased with the
         computational basis; this is not a test of a genuine complete MUB
         set (see Remark "Scope of the constructive formula").
Reference: Remark "Scope of the constructive formula" in the manuscript.
Status: VALID. Of the four additional bases, three (X, Y, and
    {X⊗Z,Z⊗X,Y⊗Y}) are mutually unbiased with the computational basis and
    saturate at machine precision (~1e-16) across r=0.3, 0.6, 0.9. The
    fourth ({X⊗Y,Y⊗X,Z⊗Z}) is NOT mutually unbiased with the computational
    basis (overlaps 0 or 1/2, not 1/4) and its failure to saturate is
    therefore not a counterexample to Theorem general-diagonal-saturation.
"""

import numpy as np
from itertools import product

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
paulis = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

def kron(a, b):
    return np.kron(a, b)

# Five selected maximal commuting sets of 2-qubit Pauli operators;
# these sets are not asserted to form a genuine complete MUB construction.
# Each set contains three commuting nontrivial Pauli operators;
# only two are algebraically independent, while the third is their product
# up to an overall phase/sign. Their joint eigenvectors form one basis.
generator_sets = [
    [kron(Z,I2), kron(I2,Z), kron(Z,Z)],   # computational basis (M1)
    [kron(X,I2), kron(I2,X), kron(X,X)],
    [kron(Y,I2), kron(I2,Y), kron(Y,Y)],
    [kron(X,Y), kron(Y,X), kron(Z,Z)],
    [kron(X,Z), kron(Z,X), kron(Y,Y)],
]

def common_eigenbasis(gens):
    # Simultaneous eigenbasis of commuting Hermitian generators
    M = sum((k+1)*10*g for k, g in enumerate(gens))  # generic combination
    evals, evecs = np.linalg.eigh(M)
    return evecs  # columns are eigenvectors

def von_neumann_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return -np.sum(ev * np.log(ev))

def M_matrix(d, r):
    n = np.arange(d)
    diff = n[:, None] - n[None, :]
    return (1.0 / d) * r ** (diff ** 2)

def basis_gap(d, r, basis_vectors):
    M = M_matrix(d, r)
    S_M = von_neumann_entropy(M)
    max_gap = 0.0
    for k in range(d):
        psi = basis_vectors[:, k]
        outer = np.outer(psi.conj(), psi)
        sigma_un = outer * M
        p = np.trace(sigma_un).real
        if p < 1e-12:
            continue
        sigma = sigma_un / p
        S_k = von_neumann_entropy(sigma)
        max_gap = max(max_gap, abs(S_k - S_M))
    return max_gap

print("=== Preliminary d=4 test with selected maximal commuting Pauli-subgroup bases ===")
d = 4
gen_names = [
    ['Z⊗I', 'I⊗Z', 'Z⊗Z'],
    ['X⊗I', 'I⊗X', 'X⊗X'],
    ['Y⊗I', 'I⊗Y', 'Y⊗Y'],
    ['X⊗Y', 'Y⊗X', 'Z⊗Z'],
    ['X⊗Z', 'Z⊗X', 'Y⊗Y'],
]
for r in [0.3, 0.6, 0.9]:
    print(f"--- r={r} ---")
    for i, gens in enumerate(generator_sets):
        basis = common_eigenbasis(gens)
        g = basis_gap(d, r, basis)
        print(f"  basis set {i} (gens: {gen_names[i]}): gap={g:.3e}")