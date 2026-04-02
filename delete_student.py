import os
import sys
import shutil
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# ────────────────────────────────────────────────
DATASET_PATH = "../dataset"
MODEL_PATH   = "model/face_recognition_knn.pkl"
# ────────────────────────────────────────────────

if len(sys.argv) < 3:
    print("❌ Usage: python delete_student.py <formatted_name> <uid>")
    sys.exit(1)

FORMATTED_NAME = sys.argv[1]   # e.g. "John_Doe"
UID            = sys.argv[2]   # Firebase UID

TARGET_FOLDER  = f"{FORMATTED_NAME} {UID}"
person_folder  = os.path.join(DATASET_PATH, TARGET_FOLDER)
LABEL_TO_REMOVE = TARGET_FOLDER   # label stored in KNN matches folder name

# ── 1. Verify folder exists ──────────────────────────────────────
if not os.path.exists(person_folder):
    print(f"⚠️  Dataset folder not found: {person_folder}")
    print("   Continuing with model update anyway…")
else:
    shutil.rmtree(person_folder)
    print(f"🗑️  Deleted dataset folder: {person_folder}")

# ── 2. Load existing model ───────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found. Nothing to update.")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    knn_clf, label_encoder = pickle.load(f)

old_classes = list(label_encoder.classes_)

if LABEL_TO_REMOVE not in old_classes:
    print(f"⚠️  Label '{LABEL_TO_REMOVE}' not found in model — nothing to remove.")
    sys.exit(0)

old_label_id = label_encoder.transform([LABEL_TO_REMOVE])[0]

X_old = np.array(knn_clf._fit_X)
y_old = np.array(knn_clf._y)

# ── 3. Filter out this student's embeddings ──────────────────────
keep_mask = y_old != old_label_id
X_filtered = X_old[keep_mask]
y_filtered_old = y_old[keep_mask]

removed = int(np.sum(~keep_mask))
print(f"🧹 Removed {removed} face embedding(s) for '{LABEL_TO_REMOVE}'")

if len(X_filtered) == 0:
    print("⚠️  No embeddings left after removal. Cannot train an empty model.")
    sys.exit(1)

# ── 4. Rebuild LabelEncoder without deleted class ────────────────
new_classes = [c for c in old_classes if c != LABEL_TO_REMOVE]

new_encoder = LabelEncoder()
new_encoder.fit(new_classes)

# Remap old numeric labels → new numeric labels
# (old label IDs may have gaps after removal; we need a clean 0..N-1 mapping)
old_label_names = label_encoder.inverse_transform(y_filtered_old)
y_new = new_encoder.transform(old_label_names)

# ── 5. Retrain KNN ───────────────────────────────────────────────
print(f"\n🧠 Retraining KNN on {len(X_filtered)} embeddings, {len(new_classes)} student(s)…")

knn_clf_new = KNeighborsClassifier(
    n_neighbors=min(3, len(new_classes)),
    algorithm="ball_tree",
    weights="distance",
)
knn_clf_new.fit(X_filtered, y_new)

# ── 6. Save updated model ─────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump((knn_clf_new, new_encoder), f)

print(f"✅ Model updated — '{LABEL_TO_REMOVE}' removed successfully.")
print(f"📦 Remaining students : {len(new_classes)}")
print(f"📦 Remaining embeddings: {len(X_filtered)}")
