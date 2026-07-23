"""
Identifier: Code S10
Purpose: Computes the local log-log slope of the spectral gap
    theta_3(0,r) - d*lambda_max(M(t)) across d=20 to d=900, for five
    values of r (0.2, 0.4, 0.6, 0.8, 0.9), to empirically characterize
    the true asymptotic convergence rate beyond the O(1/d) upper bound
    proven in Theorem "Spectral convergence rate".
Used in: Remark "Empirical convergence rate is faster than the proven
    bound", in the manuscript.
Status: VALID. Local slope converges to approximately -2 for all five
    tested r values as d increases from 20 to 900 (e.g., r=0.2: slope
    -1.93 at d=40 to -1.997 at d=900; r=0.9: slope -1.80 at d=40 to
    -1.993 at d=900), consistent with an empirical O(1/d^2) decay rate.
"""
import numpy as np
from mpmath import mp, jtheta, mpf
mp.dps = 30

r_list = [0.20, 0.40, 0.60, 0.80, 0.90]
dims = [20, 40, 80, 120, 180, 260, 400, 600, 900]

print(f"{'r':>6}{'d':>6}{'gap':>16}{'local_slope(log-log)':>24}")
for r in r_list:
    theta3 = float(jtheta(3, 0, mpf(r)))
    gaps = []
    for d in dims:
        k = np.arange(d)
        M = r**((k[:,None]-k[None,:])**2)/d
        lam_max = np.max(np.linalg.eigvalsh(M))
        gaps.append(theta3 - d*lam_max)
    for i, d in enumerate(dims):
        if i == 0:
            slope = float('nan')
        else:
            slope = (np.log(gaps[i]) - np.log(gaps[i-1])) / (np.log(dims[i]) - np.log(dims[i-1]))
        print(f"{r:6.2f}{d:6d}{gaps[i]:16.6e}{slope:24.4f}")
    print("-"*52)