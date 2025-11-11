import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Reproducible seed
rng = np.random.default_rng(12345)

# Parameters
n = 20000
d = 10000
delta = 0.05
one_minus_delta = 1 - delta

# Build indices and corresponding nonzero values for each row (vectorized)
indices = np.arange(n) % d          # values in 0..d-1
values = np.sqrt(indices + 1)       # sqrt((i mod d) + 1)

# Generate noise and observations (beta* = 0 so Y = eps)
eps = rng.standard_normal(n)
Y = eps.copy()

# Compute X^T Y via bincount (vectorized accumulation)
# For each row i: X[i,(indices[i])] = values[i], so (X^T Y)[j] = sum_{i:indices[i]==j} values[i] * Y[i]
XTY = np.bincount(indices, weights=values * Y, minlength=d)   # length d

# Compute diagonal of X^T X analytically:
# Each index j appears exactly n/d times (here 2) and contributes value^2 = j+1
count_each = n // d  # equals 2 here (since 20000/10000 = 2)
XT_X_diag = count_each * (np.arange(1, d + 1))               # (n/d) * j
# double-check (for this setup, XT_X_diag should be 2*j)
assert np.all(XT_X_diag == 2 * np.arange(1, d + 1))

# Inverse diag
XT_X_inv_diag = 1.0 / XT_X_diag

# Compute beta_hat
beta_hat = XT_X_inv_diag * XTY

# Theoretical variance for each coordinate (since eps are standard normals)
var_beta = XT_X_inv_diag  # we showed earlier var = (X^T X)^{-1}_{jj}

# Standardized statistics Z_j = beta_hat / sqrt(var_beta) should be ~ N(0,1)
Z = beta_hat / np.sqrt(var_beta)

# Threshold t corresponding to the CI: |beta_hat| <= sqrt(2 * var * log(2/delta))
t_factor = np.sqrt(2 * np.log(2 / delta))
CI_per_j = np.sqrt(2 * var_beta * np.log(2 / delta))   # same as sqrt(var) * t_factor

# Empirical counts
outside_mask = np.abs(beta_hat) > CI_per_j
n_outside = outside_mask.sum()
prop_outside = n_outside / d

# Also check standardized exceedance at z-threshold
z_threshold = t_factor
emp_prop_z_outside = np.mean(np.abs(Z) > z_threshold)

# Print diagnostics
print("Diagnostics:")
print(f"  n = {n}, d = {d}, delta = {delta}")
print(f"  Expected per-coordinate exceedance probability = {delta:.3f}")
print(f"  Threshold t (for Z) = sqrt(2*log(2/delta)) = {z_threshold:.4f}")
print(f"  Empirical number outside CI: {n_outside} / {d}  (prop = {prop_outside:.4f})")
print(f"  Empirical prop(|Z| > t) = {emp_prop_z_outside:.4f}")
print(f"  Sample mean(Z) = {Z.mean():.4e}, sample std(Z) = {Z.std(ddof=0):.4f}")
print()

# Quick binomial sanity: expected mean and std of number outside
expected_mean = d * delta
expected_std = np.sqrt(d * delta * (1 - delta))
print(f"Binomial expectation for #outside: mean={expected_mean:.1f}, std={expected_std:.2f}")

# Optional: run multiple trials to get distribution of counts (uncomment if desired)
# trials = 200
# counts = np.zeros(trials, dtype=int)
# for t in range(trials):
#     eps = rng.standard_normal(n)
#     Y = eps
#     XTY = np.bincount(indices, weights=values * Y, minlength=d)
#     beta_hat = XT_X_inv_diag * XTY
#     counts[t] = np.sum(np.abs(beta_hat) > CI_per_j)
# print("Multiple-trial summary (counts): mean=", counts.mean(), " std=", counts.std())

# Plot beta_hat and CI
j_coords = np.arange(1, d + 1)
plt.figure(figsize=(11,5))
plt.scatter(j_coords, beta_hat, s=6, alpha=0.6, label=r'$\hat\beta_j$')
plt.plot(j_coords,  CI_per_j, 'r--', linewidth=1, label='upper CI (per-j)')
plt.plot(j_coords, -CI_per_j, 'r--', linewidth=1, label='lower CI (per-j)')
plt.xlabel('coordinate j')
plt.ylabel(r'$\hat\beta_j$')
plt.title(f'OLS estimates and per-coordinate 95% CIs (emp outside = {n_outside}/{d})')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
