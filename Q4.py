import numpy as np
import itertools

# ---------------------------------------------------------
# 1. DATA TRANSCRIPTION
# ---------------------------------------------------------

# Data for Category w1 (rows are samples, columns are x1, x2, x3)
w1 = np.array([
    [ 0.42,  -0.087,  0.58],
    [-0.2,   -3.3,   -3.4],
    [ 1.3,   -0.32,   1.7],
    [ 0.39,   0.71,   0.23],
    [-1.6,   -5.3,   -0.15],
    [-0.029,  0.89,  -4.7],
    [-0.23,   1.9,    2.2],
    [ 0.27,  -0.3,   -0.87],
    [-1.9,    0.76,  -2.1],
    [ 0.87,  -1.0,   -2.6]
])

# Data for Category w2 (rows are samples, columns are x1, x2, x3)
w2 = np.array([
    [-0.4,    0.58,   0.089],
    [-0.31,   0.27,  -0.04],
    [ 0.38,   0.055, -0.035],
    [-0.15,   0.53,   0.011],
    [-0.35,   0.47,   0.034],
    [ 0.17,   0.69,   0.1],
    [-0.011,  0.55,  -0.18],
    [-0.27,   0.61,   0.12],
    [-0.065,  0.49,   0.0012],
    [-0.12,   0.054, -0.063]
])

# ---------------------------------------------------------
# 2. SOLUTION IMPLEMENTATION
# ---------------------------------------------------------

def print_separator(title):
    print(f"\n--- {title} ---")

# (a) 1D Gaussian MLE for each feature of w1
print_separator("(a) 1D MLE for w1 features")
features = ["x1", "x2", "x3"]
for i in range(3):
    feature_data = w1[:, i]
    mu_hat = np.mean(feature_data)
    # var uses ddof=0 for MLE (divide by N)
    var_hat = np.var(feature_data, ddof=0) 
    
    print(f"Feature {features[i]}:")
    print(f"  μ_hat = {mu_hat:.4f}")
    print(f"  σ²_hat = {var_hat:.4f}")

# (b) 2D Gaussian MLE for pairings of w1
print_separator("(b) 2D MLE for w1 pairings")
pairs = [(0, 1), (0, 2), (1, 2)] # (x1,x2), (x1,x3), (x2,x3)

for idx1, idx2 in pairs:
    # Extract the two columns
    data_2d = w1[:, [idx1, idx2]]
    
    mu_vec = np.mean(data_2d, axis=0)
    # rowvar=False because rows are samples. bias=True for MLE (div by N).
    sigma_mat = np.cov(data_2d, rowvar=False, bias=True)
    
    print(f"Pair ({features[idx1]}, {features[idx2]}):")
    print(f"  Mean Vector (μ): {np.round(mu_vec, 4)}")
    print(f"  Covariance Matrix (Σ):\n{np.round(sigma_mat, 4)}\n")

# (c) 3D Gaussian MLE for w1
print_separator("(c) 3D MLE for w1")
mu_vec_3d = np.mean(w1, axis=0)
sigma_mat_3d = np.cov(w1, rowvar=False, bias=True)

print(f"Mean Vector (μ): {np.round(mu_vec_3d, 4)}")
print("Covariance Matrix (Σ):")
print(np.round(sigma_mat_3d, 4))

# (d) Diagonal Assumption for w2
print_separator("(d) Diagonal MLE for w2")
# Estimate mean normally
mu_vec_w2 = np.mean(w2, axis=0)

# Estimate variance for each feature independently
variances_w2 = np.var(w2, axis=0, ddof=0)

# Create a diagonal matrix from the variances
diagonal_sigma = np.diag(variances_w2)

print(f"Mean Vector (μ) for w2: {np.round(mu_vec_w2, 4)}")
print("Diagonal Covariance Matrix (Σ) for w2:")
print(np.round(diagonal_sigma, 6))