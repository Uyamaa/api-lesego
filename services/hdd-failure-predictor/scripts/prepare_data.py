from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

df = pd.read_csv(data_dir / "2022-12-31.csv")

columns_needed = [
    "smart_194_raw",  # temperature
    "smart_9_raw",    # power_on_hours
    "smart_5_raw",    # reallocated_sectors
    "smart_10_raw",   # spin_retry_count
    "smart_184_raw",  # end_to_end_error
    "smart_187_raw",  # reported_uncorrectable
    "smart_188_raw",  # command_timeout
    "smart_197_raw",  # current_pending_sector
    "smart_198_raw",  # offline_uncorrectable
    "failure",        # the label
]
df_selected = df[columns_needed]

df_selected = df_selected.rename(columns={
    "smart_194_raw": "temperature",
    "smart_9_raw": "power_on_hours",
    "smart_5_raw": "reallocated_sectors",
    "smart_10_raw": "spin_retry_count",
    "smart_184_raw": "end_to_end_error",
    "smart_187_raw": "reported_uncorrectable",
    "smart_188_raw": "command_timeout",
    "smart_197_raw": "current_pending_sector",
    "smart_198_raw": "offline_uncorrectable",
    "failure": "failure"
})

df_clean = df_selected.dropna()

df_clean.to_csv(data_dir / "processed_data.csv", index=False)

print(f"Total rows: {len(df_clean)}")
print(f"Failures: {df_clean['failure'].sum()}")