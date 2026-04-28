import os
import sys
import cv2
import face_recognition
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -------------------------------
DATASET_PATH = "../../dataset"
MODEL_PATH = "../model/face_recognition_knn.pkl"

# Check if full_name and UID are provided
if len(sys.argv) < 3:
    print("❌ Usage: python new_student_embed.py <full_name> <student_uid>")
    sys.exit(1)

FULL_NAME = sys.argv[1]
UID = sys.argv[2]

# -------------------------------
# Find the folder matching "full_name uid"
# -------------------------------
TARGET_FOLDER_NAME = f"{FULL_NAME} {UID}"
person_folder = os.path.join(DATASET_PATH, TARGET_FOLDER_NAME)

if not os.path.exists(person_folder):
    raise FileNotFoundError(f"Could not find folder: {TARGET_FOLDER_NAME} in {DATASET_PATH}")

# We use the full folder name as the label in the classifier
LABEL_NAME = TARGET_FOLDER_NAME

# -------------------------------
# Load existing model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train initial model first.")

with open(MODEL_PATH, "rb") as f:
    knn_clf, label_encoder = pickle.load(f)

# Get old embeddings and labels
X_old = list(knn_clf._fit_X)
y_old = list(knn_clf._y)

# -------------------------------
# Load new person images
# -------------------------------
images = os.listdir(person_folder)
total_images = len(images)

X_new = []
y_new = []
faces_added = 0

print(f"\n👤 Adding new person: {LABEL_NAME}")
print(f"📸 Total images found: {total_images}\n")

for idx, img_name in enumerate(images, start=1):
    img_path = os.path.join(person_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{idx}/{total_images}] ❌ Cannot read {img_name}")
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if len(boxes) != 1:
        print(f"[{idx}/{total_images}] ⚠️ Skipped {img_name} (faces detected: {len(boxes)})")
        continue

    encoding = face_recognition.face_encodings(rgb, boxes)[0]
    X_new.append(encoding)

    # Encode new label using existing LabelEncoder
    if LABEL_NAME in label_encoder.classes_:
        label = label_encoder.transform([LABEL_NAME])[0]
    else:
        # Update classes list to include new student
        classes = list(label_encoder.classes_)
        classes.append(LABEL_NAME)
        label_encoder.classes_ = np.array(classes)
        label = label_encoder.transform([LABEL_NAME])[0]

    y_new.append(label)
    faces_added += 1

    progress = (idx / total_images) * 100
    print(f"[{idx}/{total_images} | {progress:.1f}%] ✅ {img_name} added ({faces_added} faces total)")

# -------------------------------
# Merge and Retrain
# -------------------------------
if faces_added == 0:
    print("❌ No valid faces found. Model was not updated.")
    sys.exit(1)

X_all = X_old + X_new
y_all = y_old + y_new

print("\n🧠 Retraining KNN classifier...")
knn_clf = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree", weights="distance")
knn_clf.fit(X_all, y_all)

with open(MODEL_PATH, "wb") as f:
    pickle.dump((knn_clf, label_encoder), f)

print(f"🎉 Student '{LABEL_NAME}' added successfully!")
print(f"📦 Total faces in model: {len(X_all)}")
