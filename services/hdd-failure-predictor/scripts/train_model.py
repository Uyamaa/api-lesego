"""
Trains a KNN model on real, processed Backblaze SMART data (produced by
scripts/prepare_data.py). Replaces the earlier synthetic-data version --
everything downstream (model_loader.py, main.py) is unaffected, since the
saved .pkl files have the exact same shape as before.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- 1. Load the processed data ---
# Same path pattern as prepare_data.py, so this works no matter which
# folder you run the script from.
base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "data" / "processed_data.csv"
model_dir = base_dir / "model"

df = pd.read_csv(data_path)

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

X = df[FEATURE_NAMES].values
y = df["failure"].values

# --- 2. Check class balance before doing anything else ---
# Real-world drive failure data is almost always heavily imbalanced --
# thousands of healthy days per actual failure. A KNN model trained
# on this as-is will tend to just always predict "healthy", since
# that's nearly every neighbor it will ever find.
n_total = len(y)
n_failures = int(y.sum())
n_healthy = n_total - n_failures

print(f"Total rows: {n_total}")
print(f"Healthy: {n_healthy}  |  Failures: {n_failures}")

if n_failures < 2:
    print(
        "WARNING: fewer than 2 failure examples in this dataset. "
        "A model trained on this will not meaningfully learn to detect "
        "failure at all -- consider using a Backblaze date range known "
        "to contain more failures, or combining several days' files."
    )

# --- 3. Split into train/test sets ---
# stratify=y keeps the same healthy/failure ratio in both the training
# and test sets, which matters a lot when failures are rare -- without
# it, a random split could easily put zero failures in the test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if n_failures >= 2 else None
)

# --- 4. Scale features ---
# Fit the scaler on training data only, then apply it to both sets --
# fitting on the full dataset would leak test-set information into
# training, giving a falsely optimistic sense of how well it performs.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train KNN ---
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# --- 6. Evaluate on the held-out test set ---
# This is the honest check of how the model performs on data it never
# saw during training -- not a guarantee, but far more meaningful than
# just checking it runs without errors.
y_pred = model.predict(X_test_scaled)
print("\nTest set performance:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- 7. Save both files ---
# Both MUST be loaded together at serving time -- the model alone is
# useless without the exact scaler it was trained with.
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "hdd_model.pkl")
joblib.dump(scaler, model_dir / "hdd_scaler.pkl")

print(f"\nSaved {model_dir / 'hdd_model.pkl'} and {model_dir / 'hdd_scaler.pkl'}")
print(f"Feature order (must match model_loader.py exactly): {FEATURE_NAMES}")