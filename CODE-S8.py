"""
Identifier: Code S8
Purpose: Numerically verifies Theorem (spectral asymptotic): that
    d*lambda_max(M(t)) -> theta_3(0,r(t)) = 1+2*N_infinity(r(t)) as d->inf,
    by direct eigenvalue computation of the Toeplitz matrix M(t) for
    r in {0.2,...,0.99} and d up to 900, tracking lambda_max, the spectral
    gap, von Neumann entropy, participation ratio, and condition number.
Used in: Proof/verification of Theorem "Spectral convergence rate" connecting
    the spectral structure of M(t) to the Jacobi theta function and N_infinity(r).
Status: Valid, supports Theorem (spectral-asymptotic).
"""
import numpy as np

# -----------------------------
# Parameters
# -----------------------------

r_list = [0.20,0.40,0.60,0.80,0.90,0.95,0.99]

dims = [20,40,80,120,180,260,400,600,900]

print("="*165)
print(f"{'r':>6} {'d':>6} {'lambda_max':>14} {'lambda_2':>14} {'lambda_min':>14} {'Gap':>14} {'Entropy':>14} {'Participation':>14} {'Cond.Num':>14}")
print("="*165)

for r in r_list:

    previous_lmax=None

    for d in dims:

        k=np.arange(d)

        M=r**((k[:,None]-k[None,:])**2)/d

        eig=np.linalg.eigvalsh(M)
        eig=np.sort(eig)[::-1]

        eig=eig[eig>1e-15]

        lam_max=eig[0]
        lam2=eig[1]
        lam_min=eig[-1]

        gap=lam_max-lam2

        entropy=-np.sum(eig*np.log(eig))

        participation=1/np.sum(eig**2)

        cond=lam_max/lam_min

        print(f"{r:6.2f} {d:6d} {lam_max:14.8f} {lam2:14.8f} {lam_min:14.8e} {gap:14.8f} {entropy:14.8f} {participation:14.4f} {cond:14.4e}")

    print("-"*165)

print("\n")

print("Convergence check of lambda_max")

print("="*80)

print(f"{'r':>6} {'d':>6} {'lambda_max':>15} {'Δ(lambda_max)':>18}")

print("="*80)

for r in r_list:

    prev=None

    for d in dims:

        k=np.arange(d)

        M=r**((k[:,None]-k[None,:])**2)/d

        eig=np.linalg.eigvalsh(M)

        lam=np.max(eig)

        if prev is None:
            delta=np.nan
        else:
            delta=lam-prev

        print(f"{r:6.2f} {d:6d} {lam:15.10f} {delta:18.10e}")

        prev=lam

    print("-"*80)

print("\n")

print("Sum of the largest eigenvalues")

print("="*120)

print(f"{'r':>6} {'d':>6} {'Top1':>12} {'Top2':>12} {'Top5':>12} {'Top10':>12}")

print("="*120)

for r in r_list:

    for d in dims:

        k=np.arange(d)

        M=r**((k[:,None]-k[None,:])**2)/d

        eig=np.linalg.eigvalsh(M)

        eig=np.sort(eig)[::-1]

        top1=np.sum(eig[:1])
        top2=np.sum(eig[:2])
        top5=np.sum(eig[:5])
        top10=np.sum(eig[:10])

        print(f"{r:6.2f} {d:6d} {top1:12.8f} {top2:12.8f} {top5:12.8f} {top10:12.8f}")

    print("-"*120)

print("\n")

print("\nRigorous verification of Theorem spectral-rate")
print("=" * 125)
print(
    f"{'r':>6} {'d':>6} {'d*lambda_max':>16} "
    f"{'theta_3(0,r)':>16} {'spectral gap':>16} "
    f"{'upper bound':>16} {'lower OK':>10} {'upper OK':>10}"
)
print("=" * 125)

numerical_tolerance = 1.0e-12
all_bounds_satisfied = True
maximum_lower_violation = 0.0
maximum_upper_violation = 0.0
maximum_gap_to_bound_ratio = 0.0
total_bound_checks = 0

for r in r_list:

    # theta_3(0,r) = 1 + 2*N_infinity(r)
    # The truncation at k=499 is numerically more than sufficient
    # for all tested values r <= 0.99.
    N_infinity = sum(r ** (k ** 2) for k in range(1, 500))
    theta3 = 1.0 + 2.0 * N_infinity

    for d in dims:

        indices = np.arange(d)
        M = r ** (
            (indices[:, None] - indices[None, :]) ** 2
        ) / d

        lam_max = np.max(np.linalg.eigvalsh(M))

        # Spectral gap appearing in Theorem spectral-rate.
        spectral_gap = theta3 - d * lam_max

        # Exact finite-dimensional negativity:
        # N(t,d) = sum_{k=1}^{d-1} (1-k/d) r^{k^2}.
        N_finite = sum(
            (1.0 - k / d) * r ** (k ** 2)
            for k in range(1, d)
        )

        # Right-hand side of the proved inequality:
        # 2 [N_infinity(r) - N(t,d)].
        upper_bound = 2.0 * (N_infinity - N_finite)

        lower_ok = spectral_gap >= -numerical_tolerance
        upper_ok = spectral_gap <= upper_bound + numerical_tolerance

        lower_violation = max(0.0, -spectral_gap)
        upper_violation = max(0.0, spectral_gap - upper_bound)

        maximum_lower_violation = max(
            maximum_lower_violation,
            lower_violation,
        )
        maximum_upper_violation = max(
            maximum_upper_violation,
            upper_violation,
        )

        if upper_bound > numerical_tolerance:
            gap_to_bound_ratio = spectral_gap / upper_bound
            maximum_gap_to_bound_ratio = max(
                maximum_gap_to_bound_ratio,
                gap_to_bound_ratio,
            )

        bound_ok = lower_ok and upper_ok
        all_bounds_satisfied = all_bounds_satisfied and bound_ok
        total_bound_checks += 1

        print(
            f"{r:6.2f} {d:6d} {d * lam_max:16.8f} "
            f"{theta3:16.8f} {spectral_gap:16.8e} "
            f"{upper_bound:16.8e} "
            f"{str(lower_ok):>10} {str(upper_ok):>10}"
        )

    print("-" * 125)


print("\n" + "=" * 80)
print("BOUND-VERIFICATION SUMMARY")
print("=" * 80)
print(f"Total tested (r,d) combinations: {total_bound_checks}")
print(f"All lower and upper bounds satisfied: {all_bounds_satisfied}")
print(
    "Maximum lower-bound violation: "
    f"{maximum_lower_violation:.3e}"
)
print(
    "Maximum upper-bound violation: "
    f"{maximum_upper_violation:.3e}"
)
print(
    "Maximum spectral-gap / upper-bound ratio: "
    f"{maximum_gap_to_bound_ratio:.12f}"
)

if not all_bounds_satisfied:
    raise RuntimeError(
        "At least one numerical test violates the spectral-rate bound "
        "beyond the specified tolerance."
    )