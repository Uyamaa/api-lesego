"""
Trains a small KNN model on SYNTHETIC data, matching the 9 SMART fields
your API will use. This proves the training -> saving -> loading -> serving
chain works end to end. Swap this script's data source for real Backblaze
CSVs later -- nothing else in the project needs to change.
"""

import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Generate synthetic data ---
# These 9 columns must match, in this exact order, whatever main.py sends
# to the model later. This order IS the contract between training and serving.
FEATURE_NAMES = [
    "temperature",
    "power_on_hours",
    "reallocated_sectors",
    "spin_retry_count",
    "end_to_end_error",
    "reported_uncorrectable",
    "command_timeout",
    "current_pending_sector",
    "offline_uncorrectable",
]

np.random.seed(42)
n_samples = 500

# Healthy drives: low error counts, normal temperature
healthy = np.column_stack([
    np.random.normal(35, 5, n_samples // 2),      # temperature
    np.random.uniform(0, 40000, n_samples // 2),   # power_on_hours
    np.random.poisson(1, n_samples // 2),          # reallocated_sectors
    np.random.poisson(0.5, n_samples // 2),        # spin_retry_count
    np.random.poisson(0, n_samples // 2),          # end_to_end_error
    np.random.poisson(0, n_samples // 2),          # reported_uncorrectable
    np.random.poisson(0, n_samples // 2),          # command_timeout
    np.random.poisson(1, n_samples // 2),          # current_pending_sector
    np.random.poisson(0, n_samples // 2),          # offline_uncorrectable
])
healthy_labels = np.zeros(n_samples // 2)

# Failing drives: higher temperature, higher error counts
failing = np.column_stack([
    np.random.normal(55, 8, n_samples // 2),
    np.random.uniform(20000, 80000, n_samples // 2),
    np.random.poisson(80, n_samples // 2),
    np.random.poisson(10, n_samples // 2),
    np.random.poisson(5, n_samples // 2),
    np.random.poisson(4, n_samples // 2),
    np.random.poisson(3, n_samples // 2),
    np.random.poisson(50, n_samples // 2),
    np.random.poisson(2, n_samples // 2),
])
failing_labels = np.ones(n_samples // 2)

X = np.vstack([healthy, failing])
y = np.concatenate([healthy_labels, failing_labels])

# --- 2. Scale features ---
# KNN measures distance, so every feature must be on the same scale
# (see earlier conversation: temperature 0-80 vs sectors 0-50000 problem).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Train KNN ---
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

# --- 4. Save both files ---
# Both MUST be loaded together at serving time -- the model alone is useless
# without the exact scaler it was trained with.
joblib.dump(model, "hdd_model.pkl")
joblib.dump(scaler, "hdd_scaler.pkl")

print("Saved hdd_model.pkl and hdd_scaler.pkl")
print(f"Feature order (must match model_loader.py exactly): {FEATURE_NAMES}")
