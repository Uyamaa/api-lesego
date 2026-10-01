# train_model.py

Trains the KNN model used by the `/predict` endpoint, using real processed
Backblaze SMART data instead of synthetic data.

## What it does

1. Loads `data/processed_data.csv` (produced by `scripts/prepare_data.py`)
2. Splits it into the 9 SMART feature columns (`X`) and the `failure` label (`y`)
3. Checks and prints the class balance (how many healthy vs. failing rows exist)
4. Splits into training (80%) and test (20%) sets, preserving the healthy/failure
   ratio in both (`stratify=y`)
5. Scales the features — fit on the training set only, to avoid leaking test
   data into training
6. Trains a `KNeighborsClassifier`
7. Evaluates on the held-out test set and prints a `classification_report`
8. Saves `model/hdd_model.pkl` and `model/hdd_scaler.pkl`

## Why it's structured this way

This replaces an earlier version that trained on synthetic (fake) healthy/failing
data, used only to prove the training → saving → loading → serving chain worked
end to end. This version trains on your actual Backblaze data instead — nothing
downstream (`model_loader.py`, `main.py`) needed to change, since the saved
`.pkl` files have the same shape either way.

Real-world drive failures are rare, so this script checks class balance up front
rather than silently training a model that never learns to detect failure. If
`Failures` is 0 or 1 in the printed output, you'll see a warning — that means
this particular Backblaze date range doesn't have enough failure examples to
train something meaningful, and you'd want a different date range or a combined
multi-day dataset instead.

## How to run it

From inside `scripts/`, with your virtual environment activated:

```bash
python train_model.py
```

## Reading the output

```
Total rows: <N>
Healthy: <N>  |  Failures: <N>

Test set performance:
              precision    recall  f1-score   support
           0       ...
           1       ...
```

- **Total / Healthy / Failures** — confirms how imbalanced the dataset actually is.
- **classification_report, row `1` (failure)** — this is the number that actually
  matters. High precision/recall on row `0` (healthy) is nearly automatic when
  failures are rare; row `1`'s numbers tell you whether the model learned
  anything useful about failures specifically. A recall of `0.00` on row `1`
  means the model never once correctly identified a failure in the test set —
  a strong signal more/better failure data is needed before relying on this
  model for anything real.

## After running

`model/hdd_model.pkl` and `model/hdd_scaler.pkl` are overwritten with the newly
trained versions. Restart `uvicorn` (or rebuild the Docker image) to pick up
the change — they're loaded once at startup in `model_loader.py`, not re-read
on every request.