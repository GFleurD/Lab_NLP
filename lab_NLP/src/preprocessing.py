# src/preprocessing.py

import os
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------
# 1. Load and clean CSV files
# -----------------------------

# Expert dataset (positive class)
expert_file = os.path.join(RAW_DIR, "medicheck-expert.csv")
df_expert = pd.read_csv(expert_file, skiprows=1, header=None, quotechar='"')
df_expert = df_expert.iloc[:, [0]]  # keep only first column

# Negative dataset (negative class)
neg_file = os.path.join(RAW_DIR, "medicheck-neg.csv")
with open(neg_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
df_neg = pd.DataFrame(lines)

# -----------------------------
# 2. Save cleaned CSVs for inspection
# -----------------------------
df_expert.to_csv(os.path.join(PROCESSED_DIR, "medicheck-expert_cleaned.csv"),
                 index=False, header=False, quoting=csv.QUOTE_ALL)

df_neg.to_csv(os.path.join(PROCESSED_DIR, "medicheck-neg_cleaned.csv"),
              index=False, header=False, quoting=csv.QUOTE_ALL)

print(f"Saved cleaned CSVs to {PROCESSED_DIR}")

# -----------------------------
# 3. Create training/testing split
# -----------------------------
X = pd.concat([df_expert, df_neg], ignore_index=True)[0].values
y = np.array([0]*len(df_expert) + [1]*len(df_neg))  # 0=expert, 1=negative

# Change test_size here if needed
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# -----------------------------
# 4. Save as .npy files
# -----------------------------
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
print(f"Saved .npy files to {PROCESSED_DIR}")
