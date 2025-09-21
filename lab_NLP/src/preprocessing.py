import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Medicheck Expert ---
expert_file = os.path.join(RAW_DIR, "medicheck-expert.csv")
output_expert_file = os.path.join(PROCESSED_DIR, "medicheck-expert_cleaned.csv")

# Use Python engine for more robust parsing
df_expert = pd.read_csv(expert_file, skiprows=1, header=None, engine='python', quotechar='"')

# Keep only the first column
df_expert = df_expert.iloc[:, [0]]

# Save cleaned CSV
df_expert.to_csv(output_expert_file, index=False, header=False)
print(f"Shape of cleaned expert data: {df_expert.shape}")
print(f"Processed expert file saved to {output_expert_file}")

# --- Medicheck Neg ---
neg_file = os.path.join(RAW_DIR, "medicheck-neg.csv")
output_neg_file = os.path.join(PROCESSED_DIR, "medicheck-neg_cleaned.csv")

#the negative dataset spreads across multiple rows in an irregular way, so read line by line
neg_lines = []
with open(neg_file, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()
        if stripped:
            neg_lines.append(stripped)

#convert into dataframe
df_neg_combined = pd.DataFrame(neg_lines)
df_neg_combined.to_csv(output_neg_file, index=False, header=False)
print(f"Shape of cleaned neg data: {df_neg_combined.shape}")

#creating x and y files for the test train split
# Load cleaned data
X_pos = pd.read_csv(f"{PROCESSED_DIR}/medicheck-expert_cleaned.csv", header=None).iloc[:,0].values
X_neg = pd.read_csv(f"{PROCESSED_DIR}/medicheck-neg_cleaned.csv", header=None).iloc[:,0].values

# Create labels
y_pos = np.ones(len(X_pos))
y_neg = np.zeros(len(X_neg))

# Combine
X = np.concatenate([X_pos, X_neg])
y = np.concatenate([y_pos, y_neg])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save text arrays for record if needed
np.save(f"{PROCESSED_DIR}/X_train.npy", X_train)
np.save(f"{PROCESSED_DIR}/X_test.npy", X_test)
np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
np.save(f"{PROCESSED_DIR}/y_test.npy", y_test)

