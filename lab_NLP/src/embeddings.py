import numpy as np
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "./data/processed"

# Load the train/test splits
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy", allow_pickle=True)
X_test = np.load(f"{PROCESSED_DIR}/X_test.npy", allow_pickle=True)
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # example SBERT model

# Generate embeddings
X_embeddings_train = model.encode(X_train.tolist(), show_progress_bar=True)
X_embeddings_test = model.encode(X_test.tolist(), show_progress_bar=True)

# Save embeddings for later
np.save(f"{PROCESSED_DIR}/X_embeddings_train.npy", X_embeddings_train)
np.save(f"{PROCESSED_DIR}/X_embeddings_test.npy", X_embeddings_test)
