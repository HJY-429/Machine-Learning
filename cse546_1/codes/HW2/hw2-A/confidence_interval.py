import numpy as np
import matplotlib.pyplot as plt

n = 20000
d = 10000
delta = 0.05

# Each x_i has one nonzero coordinate at index (i % d)
indices = np.arange(n) % d
values = np.sqrt(indices + 1)  # corresponding sqrt((i mod d)+1)
eps = np.random.randn(n)       # noise
Y = eps

XTY = np.zeros(d)
for i in range(n):
    j = indices[i]
    XTY[j] += values[i] * Y[i]

# Compute diagonal entries of X^T X and its inverse
q = n // d
r = n % d
j = np.arange(1, d + 1)                # j = 1,2,...,d

# indicator for extra appearance for math-indexing (j in {2,3,...,r+1})
extra = ((j >= 2) & (j <= (r + 1))).astype(int)

counts = q + extra                     # counts per coordinate j (math indexing)
XTX = counts * j
print(XTX)
# but simpler for n multiple of d (here true): 2j
# XTX = 2 * np.arange(1, d + 1)
XTX_inv = 1 / XTX
beta_hat = XTX_inv * XTY

# Confidence interval (1 - delta = 0.95)
CI = np.sqrt(2 * XTX_inv * np.log(2 / delta))

# Count how many beta_j fall outside the CI
outside = np.sum(np.abs(beta_hat) > CI)
print(f"Number of beta_j outside the confidence interval: {outside} / {d}")
print(f"rate: {outside / d * 100}%")

# Plot beta_hat with confidence intervals
plt.figure(figsize=(10, 5))
plt.scatter(np.arange(1, d + 1), beta_hat, s=8, color='green', label=r'$\widehat{\beta}_j$')
plt.plot(np.arange(1, d + 1), CI, '--', color='red', linewidth=2, label='Upper CI')
plt.plot(np.arange(1, d + 1), -CI, '--', color='purple', linewidth=2, label='Lower CI')
plt.xlabel(r'$j \in {1,2, ..., d}$')
plt.ylabel(r'$\widehat{\beta}_j$')
plt.title(r'Confidence Intervals with $\widehat{\beta}_j$')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('confidence_interval.pdf')
plt.show()